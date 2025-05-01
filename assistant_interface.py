<update_assistant_interface>

```
import logging
import io, sys, os
import json
import httpx
import re
import requests
from typing import AsyncGenerator, Callable, Awaitable, Optional, Dict, List, Tuple
from pydantic import BaseModel, Field
import asyncio
import threading
from queue import Queue
from jinja2 import Template
from datetime import datetime
from dataclasses import dataclass
from open_webui.models.files import FileForm
from open_webui.utils.misc import (
    add_or_update_user_message,
)
from open_webui.models.messages import (
    Messages,
    MessageModel,
    MessageResponse,
    MessageForm,
)
from open_webui.retrieval.vector.connector import VECTOR_DB_CLIENT
from langchain_core.documents import Document
from open_webui.models.knowledge import (
    Knowledges,
    KnowledgeForm,
    KnowledgeResponse,
    KnowledgeUserResponse,
    KnowledgeUserModel,
)
from open_webui.models.files import Files, FileForm
from hepai import HRModel
import numpy as np

sys.path.append("/Users/xuliang/third_party")
from assistant_utils.tools import (
        ManagedThread,
        ThreadManager,
        oai_chat_completion,
        transfer_userproxy_role,
        merge_adjacent_roles,
        extract_json,
        strip_triple_backtick,
)


class WorkingMemory:
    """
    Working memory class
    """
    def __init__(self, chat_id, max_size=10):
        self.filename = f"working_memory_{chat_id}.json"
        self.max_size = max_size
        self.objects: List[VectorDBResultObject] = []
        self.load()

    def add_object(self, new_object) -> bool:
        is_new = True
        # check if new_object already exists, move it to the end
        for i, obj in enumerate(self.objects):
            if obj.document == new_object.document:
                self.objects.pop(i)
                is_new = False

        self.objects.append(new_object)

        # Remove the oldest
        if len(self.objects) > self.max_size:
            self.objects.pop(0)

        return is_new

    def save(self):
        with open(self.filename, "w") as file:
            json.dump([asdict(obj) for obj in self.objects], file)

    def load(self):
        log.info("Loading working memory...")
        if os.path.exists(self.filename):
            with open(self.filename, "r") as file:
                data = json.load(file)
                self.objects = [VectorDBResultObject(**obj) for obj in data]
        else:
            log.info("No working memory file found.")

    def __str__(self):
        output = ""
        for obj in self.objects:
            source = obj.metadata.get("source", "Unknown").replace("~", "/")
            output += f'<context source="{source}">\n{obj.document}\n</context>\n\n'
        return output

    def __repr__(self):
        return str(self.objects)


class EventFlags:
    """
    Event flags class
    """
    def __init__(self):
        self.thinking_state: int = 0
        self.mem_updated = False


class SessionBuffer:
    """
    Variables for a chat session
    """
    def __init__(self, chat_id):
        self.rag_thread_mgr: ThreadManager = ThreadManager()
        self.rag_result_queue: Queue = Queue()
        self.memory: WorkingMemory = WorkingMemory(chat_id)
        self.chat_history: str = ""
        self.code_worker = None
        self.code_worker_op_system = "Linux"

class RoundBuffer:
    """
    Variables for a chat round
    """
    def __init__(self):
        self.sentence_buffer: str = ""
        self.sentences: List[str] = []
        self.window_embeddings: List = []
        self.paragraph_begin_id: int = 0
        self.total_response = ""
        self.reasoning_content = ""
        self.tools = []

    def reset(self):
        self.sentence_buffer = ""
        self.sentences = []
        self.window_embeddings = []
        self.paragraph_begin_id = 0
        self.total_response = ""
        self.reasoning_content = ""
        self.tools = []


class Assistant:
    def __init__(self, valves):
        self.model_id = "Yui-000"
        self.valves = valves
        self.data_prefix = "data: "
        self.rag_thread_max = 1

    async def interface(
        self, body: dict, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> AsyncGenerator[str, None]:
        # 验证配置
        if not self.valves.MODEL_API_KEY:
            yield json.dumps({"error": "未配置API密钥"}, ensure_ascii=False)
            return

        try:
            # ==================================================================
            # 初始化变量
            # ==================================================================
            # 获取chat id, user id, message id
            user_id, chat_id, message_id = self.extract_event_info(__event_emitter__)
    
            # 初始化知识库
            collection_name_ids = await self.init_knowledge(user_id, chat_id)
    
            event_flags: EventFlags = EventFlags()
            session_buffer: SessionBuffer = SessionBuffer(chat_id)
            round_buffer: RoundBuffer = RoundBuffer()
    
            # Initialize tools
            TOOL = {}
            prompt_templates = {}
    
            if self.valves.USE_CODE_INTERFACE:
                TOOL["code_interface"] = self.code_interface
                prompt_templates["code_interface"] = (
                    self.DEFAULT_CODE_INTERFACE_PROMPT()
                )
                session_buffer.code_worker, session_buffer.code_worker_op_system = self.init_code_worker()
            if self.valves.USE_WEB_SEARCH:
                TOOL["web_search"] = self.web_search
                prompt_templates["web_search"] = self.DEFAULT_WEB_SEARCH_PROMPT()
    
            # ==================================================================
            # 预处理消息（规范化、解析图片）
            # ==================================================================
            messages = body["messages"]
            await self.process_message_figures(messages, __event_emitter__)
            # User proxy转移到User 角色以保护身份认同
            await transfer_userproxy_role(messages)
            # 处理消息以防止相同的角色
            await merge_adjacent_roles(messages)
            # 更新系统提示词
            self.set_system_prompt(messages, prompt_templates, session_buffer)
            # RAG用户消息
            if messages[-1]["role"] == "user":
                results = await self.query_collection(
                    messages[-1]["content"], collection_name_ids, __event_emitter__
                )
                for result in results:
                    session_buffer.memory.add_object(result)
    
            log.debug("Old message:")
            log.debug(messages[1:])
    
            # ==================================================================
            # 发起API请求
            # ==================================================================
            create_new_round = True
            round_count = 0
            while create_new_round and round_count < self.valves.MAX_LOOP:
                # RAG队列大于阈值时等待，再继续下一轮
                if session_buffer.rag_thread_mgr.active_count() > self.rag_thread_max:
                    await asyncio.sleep(0.1)
                    continue
    
                # 更新系统提示词
                self.set_system_prompt(messages, prompt_templates, session_buffer)
    
                log.debug(messages[1:])
                log.info("Starting chat round")

                choices = oai_chat_completion(
                    model=self.valves.BASE_MODEL,
                    url=self.valves.MODEL_API_BASE_URL,
                    api_key=self.valves.MODEL_API_KEY,
                    body=body
                )

                async for choice in choices:
                    # ======================================================
                    # 提前结束条件判断
                    # ======================================================
                    # 检查rag结果队列
                    while not session_buffer.rag_result_queue.empty():
                        result = session_buffer.rag_result_queue.get()
                        for result_object in result:
                            session_buffer.memory.add_object(result_object)
                        event_flags.mem_updated = True
    
                    # 判断是否打断当前生成并进入下一轮
                    if self.check_refresh_round(session_buffer, event_flags):
                        break
   
                    # ======================================================
                    # 结束条件判断
                    # ======================================================
                    # 更新工具调用情况
                    round_buffer.tools = self.find_tool_usage(round_buffer.total_response)
                    early_end_round = self.check_early_end_round(round_buffer.tools)
    
                    if choice.get("finish_reason") or early_end_round:
                        create_new_round = early_end_round
                        log.info("Finishing chat")
                        self.update_assistant_message(
                            messages,
                            round_buffer,
                            event_flags,
                            prefix_reasoning=False,
                        )
                        paragraph = await self.finalize_sentences(
                            session_buffer, round_buffer
                        )
                        # =================================================
                        # Call tools
                        # =================================================
                        if round_buffer.tools:
                            yield f'\n\n<details type="status">\n<summary>Running...</summary>\nRunning\n</details>\n'
                            user_proxy_reply = ""
                            for i, tool in enumerate(round_buffer.tools):
                                if i > 0:
                                    await asyncio.sleep(0.1)
                                summary, content = await TOOL[tool["name"]](
                                    session_buffer, tool["attributes"], tool["content"]
                                )
    
                                # Check for image urls
                                image_urls = self.extract_image_urls(content)
    
                                if image_urls:
                                    figure_summary = await self.query_vision_model(
                                        self.VISION_MODEL_PROMPT(), image_urls, __event_emitter__
                                    )
                                    content += figure_summary
                                
                                user_proxy_reply += f"{summary}\n\n{content}\n\n"
                                yield f'\n<details type="user_proxy">\n<summary>{summary}</summary>\n{content}\n</details>\n'
                            # Update user proxy message
                            messages.append(
                                {
                                    "role": "user",
                                    "content": user_proxy_reply,
                                }
                            )
    
                        if paragraph:
                            session_buffer.rag_thread_mgr.submit(
                                self.query_collection_to_queue,
                                args=(
                                    session_buffer.rag_result_queue,
                                    [paragraph],
                                    collection_name_ids,
                                    __event_emitter__
                                ),
                            )
    
                        # Reset varaiables
                        round_buffer.reset()
                        round_count += 1
                        log.debug(f"Current round: {round_count}, create_new_round: {create_new_round}")
                        break
                    # ======================================================
                    # 思考状态处理
                    # ======================================================
                    state_output = await self.update_thinking_state(
                        choice.get("delta", {}), event_flags
                    )
                    if state_output:
                        yield state_output  # 直接发送状态标记
    
                    do_semantic_segmentation = False
                    if self.valves.REALTIME_RAG:
                        do_semantic_segmentation = True
                    if self.valves.REALTIME_IO and self.is_answering(
                        event_flags.thinking_state
                    ):
                        do_semantic_segmentation = True
                    # ======================================================
                    # 内容处理
                    # ======================================================
                    content = self.process_content(
                        choice["delta"], round_buffer, event_flags
                    )
                    if content:
                        yield content
                        self.update_assistant_message(
                            messages,
                            round_buffer,
                            event_flags,
                            prefix_reasoning=True,
                        )
    
                        if do_semantic_segmentation:
                            # 根据语义分割段落
                            sentence_n = await self.update_sentence_buffer(
                                content, round_buffer
                            )
                            if sentence_n == 0:
                                continue
                            new_paragraphs = await self.semantic_segmentation(
                                sentence_n, round_buffer
                            )
                            if len(new_paragraphs) == 0:
                                continue
                            # 实时RAG搜索
                            for paragraph in new_paragraphs:
                                session_buffer.rag_thread_mgr.submit(
                                    self.query_collection_to_queue,
                                    args=(
                                        session_buffer.rag_result_queue,
                                        [paragraph],
                                        collection_name_ids,
                                        __event_emitter__
                                    ),
                                )
    
                log.debug(messages[1:])
        except Exception as e:
            log.error(f"Error in assistant interface: {e}")
            yield json.dumps({"error": str(e)}, ensure_ascii=False)

    def find_tool_usage(self, content):
        tools = []
        # Define the regex pattern to match the XML tags
        pattern = re.compile(
            r"<(code_interface|web_search)\s+([^>]+)>(.*?)</\1>",
            re.DOTALL | re.MULTILINE,
        )

        # Find all matches in the content
        matches = pattern.findall(content)

        # If no matches found, return None
        if not matches:
            return []

        for match in matches:
            # Extract the tag name, attributes, and content
            tag_name = match[0]
            attributes_str = match[1]
            tag_content = match[2].strip()

            # Extract attributes into a dictionary
            attributes = {}
            for attr in attributes_str.split():
                if "=" in attr:
                    key, value = attr.split("=", 1)
                    value = value.strip("\"'")
                    attributes[key] = value

            # Return the XML information
            tools.append(
                {"name": tag_name, "attributes": attributes, "content": tag_content}
            )

        return tools

    def check_early_end_round(self, tools):
        """
        End the round early if certain tools are used
        """
        new_round = False
        for tool in tools:
            if tool["name"] == "code_interface":
                tool_type = tool["attributes"].get("type", "")
                if tool_type == "exec":
                    new_round = True
            if tool["name"] == "web_search":
                new_round = True
        return new_round

    def check_refresh_round(
        self, session_buffer: SessionBuffer, event_flags: EventFlags
    ):
        """
        break current post request and start a new one, but is still in the same round
        """
        if_break = False
        # 工作记忆更新时打断
        if event_flags.mem_updated:
            log.info("Breaking chat round: Memory updated")
            if_break = True
            event_flags.mem_updated = False
        # RAG队列长度超过阈值时打断
        if session_buffer.rag_thread_mgr.active_count() > self.rag_thread_max:
            log.info(
                f"Breaking chat round: RAG thread count {session_buffer.rag_thread_mgr.active_count()}"
            )
            if_break = True

        return if_break

    def extract_event_info(
        self, event_emitter: Callable[[dict], Awaitable[None]]
    ) -> Tuple[str, str, str]:
        """
        从事件发射器中提取用户ID、聊天ID和消息ID。
        """
        if not event_emitter or not event_emitter.__closure__:
            return None, None, None
        for cell in event_emitter.__closure__:
            if isinstance(request_info := cell.cell_contents, dict):
                user_id = request_info.get("user_id")
                chat_id = request_info.get("chat_id")
                message_id = request_info.get("message_id")
        return user_id, chat_id, message_id

    async def init_knowledge(self, user_id: str, chat_id: str) -> Dict:
        """
        初始化知识库数据，将其存储在 collection_name_ids 字典中。
        :param user_id: 用户ID
        :return: 知识库字典和消息记录文件名称
        """
        log.debug("Initializing knowledge bases")
        collection_name_ids: Dict[str, str] = {}  # 初始化知识库字典
        try:
            long_chat_collection_name = f"{self.model_id}-Chat-History-{user_id}"
            long_chat_file_name = f"chat-{self.model_id}-{user_id}-{chat_id}.txt"
            long_chat_knowledge = None
            knowledge_bases = Knowledges.get_knowledge_bases_by_user_id(
                user_id, "read"
            )  # 获取知识库
            # 获取知识库名称列表
            rag_collection_names = [
                name.strip() for name in self.valves.RAG_COLLECTION_NAMES.split(",")
            ]
            # 遍历知识库列表
            for knowledge in knowledge_bases:
                knowledge_name = knowledge.name  # 获取知识库名称
                if knowledge_name in rag_collection_names:
                    log.info(
                        f"Adding knowledge base: {knowledge_name} ({knowledge.id})"
                    )
                    collection_name_ids[knowledge_name] = (
                        knowledge.id  # 将知识库信息存储到字典中
                    )
                    #log.info(knowledge.data)  # output: {'file_ids': [...]}
                if knowledge_name == long_chat_collection_name:
                    long_chat_knowledge = knowledge

            if not long_chat_knowledge:
                log.info(
                    f"Creating long term memory knowledge base: {long_chat_collection_name}"
                )
                form_data = KnowledgeForm(
                    name=long_chat_collection_name,
                    description=f"Chat history for {self.model_id}",
                    access_control={},
                )
                long_chat_knowledge = Knowledges.insert_new_knowledge(
                    user_id, form_data
                )

            log.info(
                f"Loaded {len(collection_name_ids)} knowledge bases: {list(collection_name_ids.keys())}"
            )

        except Exception as e:
            raise Exception(f"Failed to initialize knowledge bases: {str(e)}")

        return collection_name_ids

    async def process_message_figures(self, messages, event_emitter: Callable[[dict], Awaitable[None]]):
            # 检查最后一条user消息是否包含图片
            log.debug("Checking last user message for images")
            if messages[-1]["role"] == "user":
                content = messages[-1]["content"]
                if isinstance(content, List):
                    text_content = ""
                    # 查找文字内容
                    for c in content:
                        if c.get("type", "") == "text":
                            text_content = c.get("text", "")
                            log.debug(
                                f"Found text in last user message: {text_content}"
                            )
                            break

                    # 查找图片内容
                    for c in content:
                        if c.get("type", "") == "image_url":
                            log.debug("Found image in last user message")
                            image_url = c.get("image_url", {}).get("url", "")
                            if image_url:
                                if image_url.startswith("data:image"):
                                    log.debug("Image URL is a data URL")
                                else:
                                    log.debug(f"Image URL: {image_url}")
                                # Query vision language model
                                vision_summary = await self.query_vision_model(
                                    self.VISION_MODEL_PROMPT(), [image_url], event_emitter
                                )
                                # insert to message content
                                text_content += vision_summary
                    # 替换消息
                    messages[-1]["content"] = text_content
                else:
                    image_urls = self.extract_image_urls(content)
                    if image_urls:
                        log.debug(f"Found image in last user message: {image_urls}")
                        # Call Vision Language Model
                        vision_summary = await self.query_vision_model(
                            self.VISION_MODEL_PROMPT(), image_urls, event_emitter
                        )
                        messages[-1]["content"] += vision_summary

            # 确保user message是text-only
            log.debug("Checking all user messages content format")
            for msg in messages:
                if msg["role"] == "user":
                    content = msg["content"]
                    if isinstance(content, List):
                        log.debug("Found a list of content in user message")
                        text_content = ""
                        # 查找文字内容
                        for c in content:
                            if c.get("type", "") == "text":
                                text_content = c.get("content", "")
                                log.debug(f"Found text in user message: {text_content}")
                                break

                        # 替换消息
                        log.debug("Replacing user message content")
                        msg["content"] = text_content

    def is_thinking(self, thinking_state):
        return thinking_state in [1, 2]

    def is_answering(self, thinking_state):
        return thinking_state == 0

    async def update_thinking_state(self, delta: dict, event_flags: EventFlags) -> str:
        """
        更新思考状态
        0: 未开始 / 回答中
        1: resasoning_content
        2: content为思考，process content时放入reasoning_content，并等待</think>
        3: </think>
        4: '\n\n'
        ---
        rc 0 -> 1  <think>\n\n
        rc 1
        content 1 -> 0 \n</think>\n\n
        ---
        rc 0 -> 1
        content ... 2
        content '</think>' 2 -> 3
        content '\n\n' 3-> 4
        content ... 4 -> 0
        ---
        """
        state_output = ""

        # log.debug(f"Thinking state old: {event_flags.thinking_state}, delta: {delta}")

        # 状态转换：未开始 -> 思考中
        if event_flags.thinking_state == 0 and delta.get("reasoning_content"):
            # 0 -> 1
            event_flags.thinking_state = 1
            state_output = "<think>\n\n"
        elif event_flags.thinking_state == 1 and delta.get("content"):
            # 1 -> 0
            event_flags.thinking_state = 0
            state_output = "\n</think>\n\n"
        elif event_flags.thinking_state == 2 and delta.get("content") == "</think>":
            # 2 -> 3
            event_flags.thinking_state = 3
            state_output = "\n</think>\n\n"
        elif event_flags.thinking_state == 3 and delta.get("content") == "\n\n":
            # 3 -> 4
            event_flags.thinking_state = 4
        elif event_flags.thinking_state == 4 and delta.get("content"):
            # 4 -> 0
            event_flags.thinking_state = 0

        # log.debug(f"Thinking state new: {event_flags.thinking_state}")

        return state_output

    def process_content(
        self, delta: dict, round_buffer: RoundBuffer, event_flags: EventFlags
    ) -> str:
        """直接返回处理后的内容"""
        if delta.get("reasoning_content", ""):
            reasoning_content = delta.get("reasoning_content", "")
            round_buffer.reasoning_content += reasoning_content
            return reasoning_content
        elif delta.get("content", ""):
            delta = delta.get("content", "")
            if event_flags.thinking_state == 0:
                # 回答状态时才放入回答
                round_buffer.total_response += delta
                return delta
            elif event_flags.thinking_state == 2:
                # 思考状态时放入思考内容
                round_buffer.reasoning_content += delta
                return delta
        return ""

    async def update_sentence_buffer(self, text, round_buffer: RoundBuffer) -> int:
        """
        更新句子缓冲区，将文本分割成句子并添加到句子列表中。
        :param text: 要添加到缓冲区的文本
        :return: 添加到句子列表的句子数量
        """
        round_buffer.sentence_buffer += text
        pattern = r'(?<=[。！？“”])|(?<=\n)|(?<=[.!?]\s)|(?<=")'
        splits = re.split(pattern, round_buffer.sentence_buffer)
        if len(splits) > 1:
            sentence_len_old = len(round_buffer.sentences)
            for i in range(len(splits) - 1):
                mark_pattern = r"^[\s\W_]+$"
                if (
                    re.match(mark_pattern, splits[i].strip())  # 纯符号
                    and len(round_buffer.sentences) > 0
                ):
                    round_buffer.sentences[-1] += splits[i]
                else:
                    round_buffer.sentences.append(splits[i])
            round_buffer.sentence_buffer = splits[-1]
            sentence_len_new = len(round_buffer.sentences)
            return sentence_len_new - sentence_len_old
        return 0

    def update_assistant_message(
        self,
        messages,
        round_buffer,
        event_flags: EventFlags,
        prefix_reasoning: bool = False,
    ):
        """更新助手消息"""
        if prefix_reasoning:
            if round_buffer.total_response:
                assistante_message = {
                    "role": "assistant",
                    "content": f"<think>\n\n{round_buffer.reasoning_content}\n</think>\n\n{round_buffer.total_response}",
                    "prefix": True,
                }
            else:
                assistante_message = {
                    "role": "assistant",
                    "content": f"<think>\n\n{round_buffer.reasoning_content}",
                    "prefix": True,
                }
                event_flags.thinking_state = 2
        else:
            if not round_buffer.total_response:
                log.error("No total_response, cannot update assistant message?")
                return

            assistante_message = {
                "role": "assistant",
                "content": round_buffer.total_response,
                "prefix": True,
            }

        if messages[-1]["role"] == "assistant":
            messages[-1] = assistante_message
        else:
            messages.append(assistante_message)

    # RAG
    def clean_text(self, text):
        # 去除emoji，避免embedding报错
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r"", text)  # no emoji

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """调用API获取文本嵌入"""
        embeddings = []
        headers = {"Authorization": f"Bearer {self.valves.EMBEDDING_API_KEY}"}
        try:
            response = requests.post(
                f"{self.valves.EMBEDDING_API_BASE_URL}/embeddings",
                headers=headers,
                json={
                    "model": self.valves.EMBEDDING_MODEL,
                    "input": [self.clean_text(text) for text in texts],
                },
            )
            response.raise_for_status()
            data = response.json()
            for item in data["data"]:
                if not "embedding" in item:
                    continue
                embeddings.append(item["embedding"])
        except Exception as e:
            log.error(f"Failed to get embedding for texts: {texts}\nError: {e}")
        return embeddings

    def cosine_similarity(self, a, b):
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    async def semantic_segmentation(self, sentence_n, round_buffer: RoundBuffer):
        """
        滑动窗口对语言模型输出进行语义分割
        :param sentence_n: 新增句子数量
        :param round_buffer: 轮次缓冲区
        :return: 新段落列表
        """
        new_paragraphs = []

        # 滑动窗口获取嵌入向量
        window_size = 3
        window_step = 1
        sim_threshold = 0.8

        for i in range(
            len(round_buffer.sentences) - sentence_n, len(round_buffer.sentences)
        ):
            begin_idx = i - window_size + 1
            if begin_idx < 0:
                continue

            log.debug("Sliding window embedding...")
            window = "".join(round_buffer.sentences[begin_idx : i + 1])

            if window.strip() == "":
                continue

            embeddings = await self.get_embeddings([window])
            if len(embeddings) == 0:
                log.warning(f"Failed to get embedding for window: {window}")
                continue

            round_buffer.window_embeddings.append(embeddings[0])
            if len(round_buffer.window_embeddings) > 1:
                similarity = self.cosine_similarity(
                    round_buffer.window_embeddings[-2],
                    round_buffer.window_embeddings[-1],
                )
                if similarity < sim_threshold:
                    paragraph = "".join(
                        round_buffer.sentences[
                            round_buffer.paragraph_begin_id : begin_idx
                        ]
                    )
                    new_paragraphs.append(paragraph)
                    round_buffer.paragraph_begin_id = begin_idx
                    log.debug(
                        f"New paragraph:\n {paragraph}\nNext Similarity: {similarity}"
                    )

        return new_paragraphs

    async def finalize_sentences(
        self, session_buffer: SessionBuffer, round_buffer: RoundBuffer
    ) -> str:
        """
        将缓冲区中的文本添加到句子列表中。
        :param session_buffer: 会话缓冲区
        :param round_buffer: 轮次缓冲区
        :return: paragraph
        """
        log.debug("Finalizing sentences")
        sentence_n = 0
        if round_buffer.sentence_buffer:
            sentence_len_old = len(round_buffer.sentences)
            mark_pattern = r"^[\s\W_]+$"
            if (
                re.match(mark_pattern, round_buffer.sentence_buffer.strip())  # 纯符号
                and len(round_buffer.sentences) > 0
            ):
                round_buffer.sentences[-1] += round_buffer.sentence_buffer
            else:
                round_buffer.sentences.append(round_buffer.sentence_buffer)
            round_buffer.sentence_buffer = ""
            sentence_len_new = len(round_buffer.sentences)
            sentence_n = sentence_len_new - sentence_len_old
        if len(round_buffer.sentences) - round_buffer.paragraph_begin_id > 0:
            paragraph = "".join(
                round_buffer.sentences[round_buffer.paragraph_begin_id :]
            )
            return paragraph
        return ""

    async def search_chroma_db(
        self, collection_id, embeddings: List, top_k: int, max_distance: float
    ):
        """
        Search Chroma DB
        """
        result_objects = []
        try:
            if not collection_id in VECTOR_DB_CLIENT.client.list_collections():
                log.warning(
                    f"Collection {collection_id} not found in Vector DB, maybe this is an empty collection."
                )
                return result_objects

            results = VECTOR_DB_CLIENT.search(
                collection_name=collection_id,
                vectors=embeddings,
                limit=top_k,
            )

            if not results:
                return result_objects

            for query_i in range(len(embeddings)):
                for k in range(len(results.ids[query_i])):
                    if (max_distance > 0) and (
                        results.distances[query_i][k] > max_distance
                    ):
                        continue
                    result_objects.append(
                        VectorDBResultObject(
                            id_=results.ids[query_i][k],
                            distance=results.distances[query_i][k],
                            document=results.documents[query_i][k],
                            metadata=results.metadatas[query_i][k],
                            query_embedding=embeddings[query_i],
                        )
                    )

        except Exception as e:
            log.error(f"Error searching vector db: {e}")

        return result_objects

    async def query_collection(
        self,
        query_keywords: List[str],
        collection_name_ids: Dict[str, str],
        event_emitter: Callable[[dict], Awaitable[None]] = None,
        knowledge_names: List[str] = [],
        top_k: int = 1,
        max_distance: float = 0,
    ) -> list:
        log.debug("Querying Knowledge Collection")
        embeddings = []

        query_keywords = [kwd.strip() for kwd in query_keywords if kwd.strip()]
        if not query_keywords:
            log.warning("No keywords to query")
            return []

        if self.valves.GENERATE_KEYWORDS_FROM_MODEL:

            new_knowledge_names, new_keywords = await self.generate_query_keywords(query_keywords, collection_name_ids)
            if new_keywords:
                query_keywords = new_keywords
            else:
                # if no keywords, skip search
                return []
                #log.warning("Fall back to original keywords")

            if not knowledge_names:
                knowledge_names = new_knowledge_names

        # default to all collections
        if not knowledge_names:
            knowledge_names = collection_name_ids.keys()

        if event_emitter:
            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "description": f'Searching {list(knowledge_names)}: {list(query_keywords)}',
                        "done": False,
                    },
                }
            )

        if len(query_keywords) == 0:
            log.warning("No keywords to query")
            return []
        # Generate query embedding
        log.debug(f"Generating Embeddings")
        try:
            query_embeddings = await self.get_embeddings(query_keywords)
            if len(query_embeddings) == 0:
                log.warning(f"Failed to get embedding for queries: {query_keywords}")
                return []

            # Search for each knowledge
            all_results = []
            for knowledge_name in knowledge_names:
                collection_id = collection_name_ids[knowledge_name]
                log.debug(f"Searching collection: {knowledge_name}")
                results = await self.search_chroma_db(
                    collection_id, query_embeddings, top_k, max_distance
                )
                if not results:
                    continue
                all_results.extend(results)

            log.debug(f"Search done with {len(all_results)} results")

            return all_results

        except Exception as e:
            log.error(f"Error querying collection: {e}")

        if event_emitter:
            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "description": f'Searching {list(knowledge_names)}: {list(query_keywords)}',
                        "done": True,
                        "hidden": True,
                    },
                }
            )

        return []

    def query_collection_to_queue(
        self,
        result_queue: Queue,
        query_keywords: List[str],
        collection_name_ids: Dict[str, str],
        event_emitter: Callable[[dict], Awaitable[None]] = None,
        knowledge_names: List[str] = [],
        top_k: int = 1,
        max_distance: float = 0.0,
    ):
        try:
            results = asyncio.run(
                self.query_collection(
                    query_keywords, collection_name_ids, event_emitter, knowledge_names, top_k, max_distance
                )
            )
            if results:
                log.debug(f"Put into results")
                result_queue.put(results)
        except Exception as e:
            log.error(f"Error querying collection to queue: {e}")


    async def generate_query_keywords(self, contexts: List, collection_name_ids):
        """
        Generate query keywords using llm
        """
        query_template = self.DEFAULT_QUERY_GENERATION_PROMPT()

        # Create a Jinja2 Template object
        template = Template(query_template)
        current_date = datetime.now()
        formatted_date = current_date.strftime("%Y-%m-%d")
        # if len(messages) > 7:
        #    short_messages = [messages[0]] + messages[-6:]
        # else:
        #    short_messages = messages

        # Render the template with a list of items
        replace = {
            #"CURRENT_DATE": formatted_date,
            "CONTEXTS": contexts,
            "COLLECTION_NAMES": list(collection_name_ids.keys())
        }
        query_prompt = template.render(**replace)
        task_messages = [{"role": "user", "content": query_prompt}]
        headers = {"Authorization": f"Bearer {self.valves.MODEL_API_KEY}"}
        payload = {
            "model": self.valves.TASK_MODEL,
            "messages": task_messages,
        }
        collection_names = []
        keywords = []
        try:
            response = requests.post(
                f"{self.valves.MODEL_API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                stream=False,
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("choices", [{}])[0]["message"]["content"]
            result_json = extract_json(content)
            collection_names = result_json.get("collection_names", [])
            keywords = result_json.get("queries", [])
        except Exception as e:
            log.error(f"Failed to generate query keywords: {e}")
        log.debug(f"Generated query keywords: {keywords}")
        return collection_names, keywords


    def estimate_tokens(text: str) -> int:
        return len(text) // 4  # Simple approximation

    async def archive_history(history: list, user_id: str, chat_id: str):
        content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        filename = f"chat-history-{user_id}-{chat_id}.txt"

        try:
            # Create file record
            file_form = FileForm(
                filename=filename,
                data={"content": content},
                meta={"name": filename}
            )
            file = Files.insert_file(file_form, user_id)

            # Add to long-term chat knowledge base
            collection_name = f"{self.model_id}-Chat-History-{user_id}"
            knowledge = Knowledges.get_knowledge_by_name(user_id, collection_name)
            if knowledge:
                Knowledges.add_file_to_knowledge(knowledge.id, file.id)

        except Exception as e:
            log.error(f"History archival failed: {e}")

    def set_system_prompt(
        self, messages, prompt_templates, session_buffer: SessionBuffer
    ):
        """
        ## Context
        ## Available Tools
        ## Physics Guides
        ## Task Prompt
        """
        # Context
        template_string = "## Context\n\n{{WORKING_MEMORY}}---\n"

        # Available Tools
        if len(prompt_templates) > 0:
            template_string += "\n## Available Tools\n"
            for i, (name, prompt) in enumerate(prompt_templates.items()):
                template_string += f"\n### {i+1}. {prompt}\n"

        # Physics Guides
        if self.valves.USE_DARKSHINE_GUIDE:
            template_string += self.DARKSHINE_PROMPT()

        if self.valves.USE_BESIII_GUIDE:
            template_string += self.BESIII_PROMPT()

        # Task Prompt
        template_string += self.GUIDE_PROMPT()

        # Create a Jinja2 Template object
        template = Template(template_string)
        current_date = datetime.now()
        formatted_date = current_date.strftime("%Y-%m-%d")

        # Render the template with a list of items
        context = {
            "CURRENT_DATE": formatted_date,
            "OP_SYSTEM": session_buffer.code_worker_op_system,
            "WORKING_MEMORY": str(session_buffer.memory)
        }
        result = template.render(**context)

        # Set system_prompt
        if messages[0]["role"] == "system":
            messages[0]["content"] = result
        else:
            context_message = {"role": "system", "content": result}
            messages.insert(0, context_message)

    # =========================================================================
    # Code Interface
    # =========================================================================
    def init_code_worker(self):
        try:
            code_worker = HRModel.connect(
                name=self.valves.CODE_WORKER_NAME,
                base_url=self.valves.CODE_WORKER_BASE_URL,
            )
            funcs = code_worker.functions()  # Get all remote callable functions.
            log.info(f"Remote callable funcs: {funcs}")
            op_system = code_worker.inspect_system()

            return code_worker, op_system

        except Exception as e:
            log.error(f"Error initializing code worker: {e}")

        return None, "Linux"

    async def code_interface(self, session_buffer, attributes: dict, content: str) -> Tuple[str, str]:
        log.debug("Starting Code Interface")
        if session_buffer.code_worker is None:
            session_buffer.code_worker, session_buffer.code_worker_op_system = self.init_code_worker()

        # Extract the code interface type and language
        code_type = attributes.get("type", "")
        lang = attributes.get("lang", "")
        filename = attributes.get("filename", "")
        content = strip_triple_backtick(content)

        if code_type == "exec":
            # Execute the code
            if filename:
                try:
                    result = session_buffer.code_worker.write_code(
                        file_path=filename,
                        content=content,
                        execute=True,
                        lang=lang,
                        timeout=-1,
                    )
                    return f"Executed code: {filename}", result
                except Exception as e:
                    return f"Error executing {filename}", f"{str(e)}"
            elif lang == "bash":
                try:
                    result = session_buffer.code_worker.run_command(command=content, timeout=300)
                    return "Command executed", result
                except Exception as e:
                    return "Error executing bash command", f"{str(e)}"
            else:
                return (
                    "No filename provided for code execution",
                    "Please provide filename in xml attribute.",
                )

        elif code_type == "write":
            if not filename:
                return (
                    "No filename provided for code writing",
                    "Please provide filename in xml attribute.",
                )
            # Write the code to a file
            try:
                result = session_buffer.code_worker.write_code(
                    file_path=filename, content=content
                )
                return f"Written file: {filename}", result
            except Exception as e:
                return f"Error writing {filename}", f"{str(e)}"

        elif code_type == "search_replace":
            if not filename:
                return (
                    "No filename provided for code search and replace",
                    "Please provide filename in xml attribute.",
                )
            # extract the original and updated code
            edit_block_pattern = re.compile(
                r"<<<<<<< ORIGINAL\s*(?P<original>.*?)"
                r"=======\s*(?P<mid>.*?)"
                r"\s*(?P<updated>.*)>>>>>>> ",
                re.DOTALL,
            )
            match = edit_block_pattern.search(content)
            if match:
                original = match.group("original")
                updated = match.group("updated")
                try:
                    result = session_buffer.code_worker.search_replace(
                        file_path=filename, original=original, updated=updated
                    )
                    return f"Updated {filename}", result
                except Exception as e:
                    return f"Error searching and replacing {filename}", f"{str(e)}"
            else:
                return (
                    "Invalid search and replace format",
                    "Format: <<<<<<< ORIGINAL\nOriginal code\n=======\nUpdated code\n>>>>>>> UPDATED",
                )
        else:
            return (
                f"Invalid code interface type `{code_type}`",
                "Available types: `exec`, `write`",
            )

    # =========================================================================
    # Web Search
    # =========================================================================

    async def _google_search(self, search_query: str) -> Tuple[List[str], List[str]]:
        """Perform Google search for a single query."""
        google_search_url = f"https://www.googleapis.com/customsearch/v1?q={search_query}&key={self.valves.GOOGLE_PSE_API_KEY}&cx={self.valves.GOOGLE_PSE_ENGINE_ID}&num=5"
        try:
            async with httpx.AsyncClient(http2=True) as client:
                response = await client.get(google_search_url)
                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items", [])
                    search_results = []
                    urls = []
                    for item in items:
                        title = item.get("title", "No title")
                        link = item.get("link", "No link")
                        urls.append(link)
                        snippet = item.get("snippet", "No snippet")
                        search_results.append(f"**{title}**\n{snippet}\n{link}\n")
                    return search_results, urls
                else:
                    return [f"Google search failed with status code {response.status_code} for query: {search_query}"], []
        except Exception as e:
            return [f"Error during Google search for query: {search_query}. Error: {str(e)}"], []

    async def _arxiv_search(self, search_query: str) -> Tuple[List[str], List[str]]:
        """Perform ArXiv search for a single query."""
        arxiv_search_url = f"http://export.arxiv.org/api/query?search_query=all:{search_query}&start=0&max_results=5"
        try:
            async with httpx.AsyncClient(http2=True) as client:
                response = await client.get(arxiv_search_url)
                if response.status_code == 200:
                    data = response.text
                    pattern = re.compile(r"<entry>(.*?)</entry>", re.DOTALL)
                    matches = pattern.findall(data)
                    arxiv_results = []
                    urls = []
                    for match in matches:
                        title_match = re.search(r"<title>(.*?)</title>", match)
                        link_match = re.search(r"<id>(.*?)</id>", match)
                        summary_match = re.search(r"<summary>(.*?)</summary>", match, re.DOTALL)
                        if title_match and link_match and summary_match:
                            title = title_match.group(1)
                            link = link_match.group(1)
                            urls.append(link)
                            summary = summary_match.group(1).strip()
                            arxiv_results.append(f"**{title}**\n{summary}\n{link}\n")
                        else:
                            log.error("Error parsing ArXiv entry.")
                    return arxiv_results, urls
                else:
                    return [f"ArXiv search failed with status code {response.status_code} for query: {search_query}"], []
        except Exception as e:
            return [f"Error during ArXiv search for query: {search_query}. Error: {str(e)}"], []

    async def web_search(self, session_buffer, attributes: dict, content: str) -> Tuple[str, str]:
        """
        :param session_buffer: 会话缓冲区
        :param attributes: 属性
        :param content: 内容
        :return: 搜索结果
        """
        log.debug("Starting Web Search")
        # Split content into multiple search queries
        search_queries = [query.strip() for query in content.splitlines() if query.strip()]
        if not search_queries:
            return "No search queries provided", ""

        engine = attributes.get("engine", "")
        all_results = []
        all_urls = []

        # Perform searches in parallel
        if engine == "google":
            tasks = [self._google_search(query) for query in search_queries]
        elif engine == "arxiv":
            tasks = [self._arxiv_search(query) for query in search_queries]
        else:
            return (
                "Invalid search source or query",
                f"Search engine: {engine}\nQueries: {search_queries}",
            )

        # Gather results from all tasks
        results = await asyncio.gather(*tasks)
        for result, urls in results:
            all_results.extend(result)
            all_urls.extend(urls)

        if all_results:
            result = "\n\n".join(all_results)
            return f"Searched {len(all_urls)} items across {len(search_queries)} queries", result
        else:
            return "No results found for any queries", "\n".join(search_queries)

    # =========================================================================
    # Vision Language Model
    # =========================================================================

    async def generate_vl_response(
        self,
        prompt: str,
        image_url: str,
        model: str = "Qwen/Qwen2-VL-72B-Instruct",
        url: str = "https://api.siliconflow.cn/v1",
        key: str = "",
    ) -> str:
        try:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url, "detail": "high"},
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                "stream": False,
                "max_tokens": 512,
                "stop": None,
                "temperature": 0.1,
                "top_p": 0.5,
                "top_k": 30,
                "frequency_penalty": 1.1,
                "n": 1,
                "response_format": {"type": "text"},
            }

            response = requests.request(
               "POST",
               url=f"{url}/chat/completions",
               json=payload,
               headers={
                   "Authorization": f"Bearer {key}",
                   "Content-Type": "application/json"
               },
               #proxies = {
               #   'http': 'http://127.0.0.1:7890',
               #   'https': 'http://127.0.0.1:7890',
               #}
            )

            # Check for valid response
            response.raise_for_status()

            # Parse and return embeddings if available
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            log.error(
                f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
            )
            return f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
        except httpx.ReadTimeout as e:
            log.error(f"Read Timeout error occurred")
            return f"Read Timeout error occurred"

        return ""

    def extract_image_urls(self, text: str) -> list:
        """
        Extract image URLs from text with 2 criteria:
        1. URLs ending with .png/.jpeg/.jpg/.gif/.svg (case insensitive)
        2. URLs in markdown image format regardless of extension

        Args:
            text: Input text containing potential image URLs

        Returns:
            List of unique image URLs sorted by first occurrence
        """
        # Match URLs with image extensions (including query parameters)
        ext_pattern = re.compile(
            r"https?:\/\/[^\s]+?\.(?:png|jpe?g|gif|svg)(?:\?[^\s]*)?(?=\s|$)",
            re.IGNORECASE,
        )

        # Match markdown image syntax URLs
        md_pattern = re.compile(r"!\[[^\]]*\]\((https?:\/\/[^\s\)]+)")

        # Find all matches while preserving order
        seen = set()
        result = []

        for match in ext_pattern.findall(text) + md_pattern.findall(text):
            if match not in seen:
                seen.add(match)
                result.append(match)

        return result

    async def query_vision_model(
        self,
        prompt: str,
        image_urls: List[str],
        event_emitter: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        if event_emitter:
            await event_emitter(
                        {
                            "type": "status", # We set the type here
                            "data": {"description": "Using Vision Model", "done": False, "hidden": False},
                        }
                    )
        # Batch logging directory-style URLs first
        for idx, url in enumerate(image_urls, 1):
            if not url.startswith("data:image"):
                log.debug(f"Processing image {idx}: {url}")
    
        # Configure execution parameters
        BATCH_SIZE = 5  # Controlled concurrency for large image batches
        results = []
        
        # Process in parallel batches
        for i in range(0, len(image_urls), BATCH_SIZE):
            batch_urls = image_urls[i:i+BATCH_SIZE]
            tasks = [
                self.generate_vl_response(
                    prompt=prompt,
                    image_url=url,
                    model="ark/doubao-vision-pro",
                    url=self.valves.MODEL_API_BASE_URL,
                    key=self.valves.MODEL_API_KEY,
                )
                for url in batch_urls
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        if event_emitter:
            await event_emitter(
                        {
                            "type": "status", # We set the type here
                            "data": {"description": "Using Vision Model", "done": True, "hidden": True},
                        }
                    )
        # Format ordered response
        return "\n\n".join(
            f"**Figure {idx}:** {res}" 
            for idx, res in enumerate(results, 1)
        )

    # =========================================================================
    # System prompt templates for Main Model
    # =========================================================================

    def DEFAULT_CODE_INTERFACE_PROMPT(self):
        return """Code Interface

You have access to a user's {{OP_SYSTEM}} computer workspace. You use `<code_interface>` XML tag to write codes to do analysis, calculations, or problem-solving.

#### Examples

User: plot something
Assistant: <code_interface type="exec" lang="python" filename="plot.py">

```python
# plot and save png figure to a relative path
```

</code_interface>

---

User: Create and test a simple cmake project named HelloWorld
Assistant: <code_interface type="write" lang="cmake" filename="HelloWorld/CMakeList.txt">

```cmake
...
```

</code_interface>

<code_interface type="write" lang="cpp" filename="HelloWorld/src/main.cpp">

```cpp
...
```

</code_interface>

<code_interface type="exec" lang="bash" filename="HelloWorld/build_and_test.sh">

```bash
#!/bin/bash
# assume run in parent directory of filename
mkdir -p build
cd build
cmake ..
make
./MyExecutable
```

</code_interface>

#### Tool Attributes

- `type`: Specifies the action to perform.
   - `exec`: Write code and execute the code immediately.
      - Supported languages: `python`, `bash`, `root` (root macro), `boss`
   - `write`: Simply write to file.
      - Supports any programming language.

- `filename`: The file path where the code will be written.  
   - Must be **relative to the user's workspace base directory**, do not use paths relative to subdirectory.

#### Usage Instructions

- The Python code you write can incorporate a wide array of libraries, handle data manipulation or visualization, perform API calls for web-related tasks, or tackle virtually any computational challenge. Use this flexibility to **think outside the box, craft elegant solutions, and harness Python's full potential**.
- An **extra line break** is always needed **between the `<code_interface>` XML tag and markdown code block**.
- Use the `<code_interface>` XML node and stop right away to wait for user's action.
- Only one code block is allowd in one `<code_interface>` XML node. DO NOT use two or more markdown code blocks together.
- Coding style instruction:
  - **Always aim to give meaningful outputs** (e.g., results, tables, summaries, or visuals) to better interpret and verify the findings. Avoid relying on implicit outputs; prioritize explicit and clear print statements so the results are effectively communicated to the user.
   - Run in batch mode. Save figures to png.
   - Prefer object-oriented programming
   - Prefer arguments with default value than hard coded
   - For potentially time-consuming code, e.g., loading file with unknown size, use argument to control the running scale, and defaulty run on small scale test.
"""

    def DEFAULT_WEB_SEARCH_PROMPT(self):
        return """Web Search

- You have access to internet, use `<web_search>` XML tag to search the web for new information and references. Example:

<web_search engine="google">
first query
second query
</web_search>

#### Tool Attributes

- `engine`: available options:
  - `google`: Search on google.
  - `arxiv`: Always use english keywords for arxiv.

####  Usage Instructions

- Err on the side of suggesting search queries if there is **any chance** they might provide useful or updated information.
- Always prioritize providing actionable and broad query that maximize informational coverage.
- Be concise and focused on composing high-quality search query, **avoiding unnecessary elaboration, commentary, or assumptions**.
- **The date today is: {{CURRENT_DATE}}**. So you can search for web to get information up do date {{CURRENT_DATE}}.
"""


    def DARKSHINE_PROMPT(self):
        return """
## DarkSHINE Physics Analysis Guide:

### Introduction

DarkSHINE Experiment is a fixed-target experiment to search for dark photons (A') produced in 8 GeV electron-on-target (EOT) collisions. The experiment is designed to detect the invisible decay of dark photons, which escape the detector with missing energy and missing momentum. The DarkSHINE detector consists of Tagging Tracker, Target, Recoil Tracker, Electromagnetic Calorimeter (ECAL), Hadronic Calorimeter (HCAL).

The Target is a thin plate (~350 um) of Tungsten.

Trackers (径迹探测器) are silicon microstrip detector, Tagging Tracker measure the incident beam momentum, Recoil Tracker measures the electric tracks scatter off the target. Missing momentum can be calculated by TagTrk2_pp[0] - RecTrk2_pp[0]

ECAL (电磁量能器) is cubics of LYSO crystal scintillator cells, with high energy precision.

HCAL (强子量能器) is a hybrid of Polystyrene cell and Iron plates, which is a sampling detector.

Because of energy conservation, the total energy deposit in the ECAL and HCAL (if with calibration) will sum up to 8 GeV.

Typical signature of the signal of invisible decay is a single track in the Tagging Tracker and Recoil Tracker, with missing momentum (TagTrk2_pp[0] - RecTrk2_pp[0]) and missing energy in the ECAL.

Bremstruhlung events results in missing momentum, but small missing energy in the ECAL.

Usually SM electron-nuclear or photon-nuclear process will create multiple tracks in the recoil tracker, thus not mis identified as signal, but still are a ratio of events passing the track number selection, and with MIP particles in the final states, becoming background. They can be veto by the HCAL with a HCAL Max Cell Energy cut (signal region defined by HCAL Max Cell energy lower than some value e.g. 1 MeV).

Process with neutrino will be irreducible background, however with ignorable branching ratio.

### Simulation and Reconstruction

#### Examples

User: For DarkSHINE, simulate and reconstruct inclusive background events
Assistant: <code_interface type="exec" lang="bash" filename="background_inclusive_eot.sh">

```bash
#!/bin/bash

# Set the original config file directory
dsimu_script_dir="/opt/darkshine-simulation/source/DP_simu/scripts"
default_yaml="$dsimu_script_dir/default.yaml"
magnet_file="$dsimu_script_dir/magnet_0.75_20240521.root"

echo "-- Preparing simulation config"
sed "s:  mag_field_input\::  mag_field_input\: \"${magnet_file}\"  \#:" $default_yaml > default.yaml

echo "-- Running simulation and output to dp_simu.root"
DSimu -y default.yaml -b 100 -f dp_simu.root > simu.out 2> simu.err

echo "-- Preparing reconstruction config (default input dp_simu.root and output dp_ana.root)"
DAna -x > config.txt

echo "-- Running reconstruction and output to dp_ana.root"
DAna -c config.txt

echo "All done!"
```

</code_interface>

#### Simulation and Reconstruction Steps

1. Configure the beam parameters and detector geometries for the simulation setup
2. Signal simulation and reconstruction
   1. Decide the free parameters to scan according to the signal model
   2. Simulate signal events
      1. Prepare config file
      2. Run simulation program
         - DSimu: DarkSHINE MC event generator
         - boss.exe: BESIII MC event generator
   3. Reconstruct the signal events.
      1. Prepare config file
      2. Run reconstruction program
         - DAna: DarkSHINE reconstruction program
         - boss.exe: BESIII reconstruction program
3. Background simulation and reconstruction
   1. Configure the physics process for background events
   2. Simulate background events
   3. Reconstruct background events

### Validation

#### Examples

User: Compare varaibles of signal and background events
Assistant: <code_interface type="exec" lang="python" filename="compare_kinematics.py">

```python
import ROOT
import numpy
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
...

def compare(column: str, fig_name: str):
    # create output dir if not exists
    # load files
    # draw histogram with pre_selection and column
    # overlay histograms of signal and background
    # save to png

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare kinematics of signal and background events.')
    parser.add_argument('--pre-selection', default='', help='Pre-selection to apply')
    parser.add_argument('--log-scale', action='store_true', help='Use log scale for y-axis')
    parser.add_argument('--signal-dir', default='eot/signal/invisible/mAp_100/dp_ana', help='Directory containing signal ROOT files')
    parser.add_argument('--background-dir', default='eot/background/inclusive/dp_ana', help='Directory containing background ROOT files')
    parser.add_argument('--out-dir', default='plots/png', help='Output directory for plots')
    args = parser.parse_args()
    
    # Loop for kinematic variables, save png with distinctable filename

```

</code_interface>

#### Validation Guide

Plot histograms to compare the signal and background kinematic distributions

#### Kinematic Variables

Tree Name: `dp`

| Column Name | Type | Description |
| --- | --- | --- |
| TagTrk2_pp | Double_t[] | Reconstructed Tagging Tracker momentum [MeV]. TagTrk2_pp[0] - Leading momentum track |
| TagTrk2_track_No | Int_t | Number of reconstructed Tagging Tracker tracks |
| RecTrk2_pp | Double_t[] | Reconstructed Recoil Tracker momentum [MeV]. RecTrk2_pp[0] - Leading momentum track |
| RecTrk2_track_No | Int_t | Number of reconstructed Recoil Tracker Tracks |
| ECAL_E_total | vector<double> | Total energy deposited in the ECAL [MeV]. ECAL_E_total[0] - Truth total energy. ECAL_E_total[1] - Smeard total energy with configuration 1. |
| ECAL_E_max | vector<double> | Maximum energy deposited of the ECAL Cell [MeV]. ECAL_E_max[0] - Truth maximum energy. ECAL_E_max[1] - Smeard maximum energy with configuration 1. |
| HCAL_E_total | vector<double> | Total energy deposited in the HCAL [MeV]. HCAL_E_total[0] - Truth total energy. HCAL_E_total[1] - Smeard total energy with configuration 1. |
| HCAL_E_Max_Cell | vector<double> | Maximum energy deposited of the HCAL Cell [MeV]. HCAL_E_Max_Cell[0] - Truth maximum energy. HCAL_E_Max_Cell[1] - Smeard maximum energy with configuration 1. |

### Cut-based Analysis

#### Examples

User: Optimize cut of `ECAL_E_total[0]` with 1 track cut.
Assistant: <code_interface type="exec" lang="python" filename="optimize_cut.py">

```python
import ROOT
import numpy
import matplotlib.pyplot as plt
import argparse
...

def optimize_cut():
    # Load files
    ...

    hist_sig = ROOT.TH1F("hist_sig", "", nbins, xmin, xmax)
    hist_bkg = ROOT.TH1F("hist_bkg", "", nbins, xmin, xmax)

    chain_sig.Draw(f"{cut_var} >> hist_sig", pre_cut)
    chain_bkg.Draw(f"{cut_var} >> hist_bkg", pre_cut)

    # Integral to a direction
    for i in range(nbins, 0, -1):
        cut_val =  hist_sig.GetBinLowEdge(i)
        s = hist_sig.Integral(i, nbins)
        b = hist_bkg.Integral(i, nbins)
        # Calculate `S/sqrt(S+B)` for each cut_val
        ...

    # Print the cut value, cut efficiency and significance for the optimized cut
    ...

    # Plot S/sqrt(S+B) vs cut value and the maximum, with clear syle, save to png with distinctble filename
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize cut value.')
    parser.add_argument('cut-var', nargs='?', default='ECAL_E_total[0]', help='Cut variable to optimize')
    parser.add_argument('--pre-cut', default='...', help='Cuts applied befor current cut var')
    parser.add_argument('--signal-dir', default='eot/signal/invisible/mAp_100/dp_ana', help='Directory containing signal ROOT files')
    parser.add_argument('--background-dir', default='eot/background/inclusive/dp_ana', help='Directory containing background ROOT files')
    args = parser.parse_args()
    
    # Optimize cut

```

</code_interface>

#### Cut-based Analysis Steps

1. Define signal region according to physics knowledge
2. Decide an initial loose cut values for signal region
3. Optimize cuts to maximize significance
4. Draw and print cutflow
5. Recursively optimize cut until the significance is maximized
   - Vary signal region definition and cut values
   - Optimize cuts to maximize significance
   - Draw and print cutflow

#### Guidelines

- If exists multiple signal regions, signal regions should be orthogonal to each other
- To scan S/sqrt(S+B), please use histogram integral in the loop, which is fast. DO NOT use GetEntries(cut) in a loop, which is extremly slow.
- Plot using matplotlib, not TGraph.
"""

    def BESIII_PROMPT(self):
        return ""

    def GUIDE_PROMPT(self):
        return """
## Task:

- You are a independent, patient, careful and accurate assistant, utilizing tools to help user. You analysis the chat history, decide and determine wether to use tool, or simply response to user. You can call tools by using xml node. Available Tools: Code Interface, Web Search, or Knowledge Search.

## Guidelines:

- Analyse the chat history to see if there are any question or task left that are waiting to be solved. Then utilizing tools to solve it.
- Check if previous tool is finished succesfully, if not, solve it by refine and retry the tool.
- If there are anything unclear, unexpected, or require validation, make it clear by iteratively use tool, until everything is clear with it's own reference (from tool). **DO NOT make ANY assumptions, DO NOT make-up any reply, DO NOT turn to user for information**.
- Always aim to deliver meaningful insights, iterating if necessary.
- All responses should be communicated in the chat's primary language, ensuring seamless understanding.
"""

    # =========================================================================
    # Prompts for task model, vision model
    # ========================================================================= 

    def DEFAULT_QUERY_GENERATION_PROMPT(self):
        return """### Task:
Analyze the context to determine the necessity of generating search queries, in the given language. By default, **prioritize generating 1-3 broad and relevant search queries** unless it is absolutely certain that no additional information is required. The aim is to retrieve comprehensive, updated, and valuable information even with minimal uncertainty. If no search is unequivocally needed, return an empty list.

### Guidelines:
- Respond **EXCLUSIVELY** with a JSON object. Any form of extra commentary, explanation, or additional text is strictly prohibited.
- Available collection names: {{COLLECTION_NAMES}}
- When generating search queries, respond in the format: { "collection_names": ["CollectionName"], "queries": ["query1", "query2"] }, ensuring each query is distinct, concise, and relevant to the topic and ensure each collection name is possibly relevant.
- If and only if it is entirely certain that no useful results can be retrieved by a search, return: { "queries": [] }.
- If not sure which collection to search, return: { "collection_names": [] }.
- Err on the side of suggesting search queries if there is **any chance** they might provide useful or updated information.
- Be concise and focused on composing high-quality search queries, avoiding unnecessary elaboration, commentary, or assumptions.
- Always prioritize providing actionable and broad queries that maximize informational coverage.

### Output:
Strictly return in JSON format: 
{
  "collection_names": ["Collection A", "Collection B"],
  "queries": ["query1", "query2", "query3"]
}

### Contexts
{% for item in CONTEXTS %}
{{ item }}
{% endfor %}
"""

    def VISION_MODEL_PROMPT(self):
        return """Please briefly explain this figure."""
```

</update_assistant_interface>
