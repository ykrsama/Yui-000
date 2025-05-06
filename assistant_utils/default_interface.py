import logging
import io, sys, os
import json
import uuid

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
    VectorDBResultObject,
    ManagedThread,
    ThreadManager,
    oai_chat_completion,
    extract_json,
    strip_triple_backtick,
    search_chroma_db,
    generate_query_keywords,
    get_embeddings,
    generate_vl_response,
    extract_image_urls,
    transfer_userproxy_role,
    merge_adjacent_roles,
)
from assistant_utils.prompt import (
    DEFAULT_CODE_INTERFACE_PROMPT,
    DEFAULT_WEB_SEARCH_PROMPT,
    DARKSHINE_PROMPT,
    BESIII_PROMPT,
    GUIDE_PROMPT,
    DEFAULT_QUERY_GENERATION_PROMPT,
    VISION_MODEL_PROMPT,
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
        self.mem_updated = False


class SessionBuffer:
    """
    Variables for a chat session
    """
    def __init__(self, chat_id):
        self.rag_thread_mgr: ThreadManager = ThreadManager()
        self.rag_result_queue: Queue = Queue()
        self.memory: WorkingMemory = WorkingMemory(chat_id)
        self.chat_id = chat_id
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
        self.prefix_mode = False
        self.tools = []

    def reset(self):
        self.sentence_buffer = ""
        self.sentences = []
        self.window_embeddings = []
        self.paragraph_begin_id = 0
        self.total_response = ""
        self.prefix_mode = False
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
        try:
            # Initialize
            user_id, chat_id, message_id = self.extract_event_info(__event_emitter__)
            if not chat_id:
                chat_id = str(uuid.uuid4())
            collection_name_ids = await self.init_knowledge(user_id, chat_id)
            event_flags: EventFlags = EventFlags()
            session_buffer: SessionBuffer = SessionBuffer(chat_id)
            round_buffer: RoundBuffer = RoundBuffer()
            TOOL = {}
            prompt_templates = {}
    
            if self.valves.USE_CODE_INTERFACE:
                TOOL["code_interface"] = self.code_interface
                prompt_templates["code_interface"] = (
                    DEFAULT_CODE_INTERFACE_PROMPT()
                )
                session_buffer.code_worker, session_buffer.code_worker_op_system = self.init_code_worker()
            if self.valves.USE_WEB_SEARCH:
                TOOL["web_search"] = self.web_search
                prompt_templates["web_search"] = DEFAULT_WEB_SEARCH_PROMPT()
    
            # Pre-process message (format and figures)
            messages = body["messages"]
            await self.process_message_figures(messages, __event_emitter__)
            await transfer_userproxy_role(messages)
            await merge_adjacent_roles(messages)
            self.set_system_prompt(messages, prompt_templates, session_buffer)

            # RAG
            if messages[-1]["role"] == "user":
                results = await self.query_collection(
                    messages[-1]["content"], collection_name_ids, __event_emitter__
                )
                for result in results:
                    session_buffer.memory.add_object(result)
    
            log.debug("Old message:")
            log.debug(messages)
    
            create_new_round = True
            round_count = 0
            while create_new_round and round_count < self.valves.MAX_LOOP:
                # Wait if RAG thread longer than threshold
                if session_buffer.rag_thread_mgr.active_count() > self.rag_thread_max:
                    await asyncio.sleep(0.1)
                    continue
    
                # Update system prompt again
                self.set_system_prompt(messages, prompt_templates, session_buffer)
    
                log.info(f"Starting chat round {round_count+1}")

                choices_stream = oai_chat_completion(
                    model=self.valves.BASE_MODEL,
                    url=self.valves.MODEL_API_BASE_URL,
                    api_key=self.valves.MODEL_API_KEY,
                    body=body
                )

                create_new_round = False
                finishing_chat = True

                async for choices in choices_stream:
                    if "error" in choices:
                        yield json.dumps({"error": choices["error"]}, ensure_ascii=False)
                        create_new_round = False
                        finishing_chat = False
                        break

                    # Check rag queue
                    while not session_buffer.rag_result_queue.empty():
                        result = session_buffer.rag_result_queue.get()
                        for result_object in result:
                            session_buffer.memory.add_object(result_object)
                        event_flags.mem_updated = True
    
                    # Check if break current chat round in the middle of generation
                    if self.check_refresh_round(session_buffer, event_flags):
                        log.debug("Breaking chat round")
                        create_new_round = True
                        finishing_chat = False
                        round_buffer.prefix_mode = True
                        break
   
                    # Check tool usage
                    round_buffer.tools = self.find_tool_usage(round_buffer.total_response)
                    early_end_round = self.check_early_end_round(round_buffer.tools)

                    # Finish reason condition
                    if choices.get("finish_reason") or early_end_round:
                        finishing_chat = True
                        break
   
                    do_semantic_segmentation = False
                    if self.valves.REALTIME_RAG:
                        do_semantic_segmentation = True
                    if self.valves.REALTIME_IO and self.is_answering(round_buffer.total_response):
                        do_semantic_segmentation = True

                    # Process content
                    content = self.process_content(
                        choices.get("delta", {}), round_buffer, event_flags
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
                            # RAG
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
                    finishing_chat = True

                if finishing_chat:
                    log.info("Finishing chat")
                    round_buffer.tools = self.find_tool_usage(round_buffer.total_response)
                    # close think tag if not closed
                    if round_buffer.total_response.startswith(f"{chr(0x003C)}think{chr(0x003E)}"):
                        if f"\n{chr(0x003C)}/think{chr(0x003E)}\n\n" not in round_buffer.total_response:
                            round_buffer.total_response += f"\n{chr(0x003C)}/think{chr(0x003E)}\n\n"
                            yield f"\n{chr(0x003C)}/think{chr(0x003E)}\n\n"

                    self.update_assistant_message(
                        messages,
                        round_buffer,
                        event_flags,
                        prefix_reasoning=False,
                    )

                    paragraph = await self.finalize_sentences(
                        session_buffer, round_buffer
                    )

                    # Call tools
                    if round_buffer.tools:
                        create_new_round = True
                        yield f'\n\n<details type="status">\n<summary>Running...</summary>\nRunning\n</details>\n'
                        user_proxy_reply = ""
                        for i, tool in enumerate(round_buffer.tools):
                            if i > 0:
                                await asyncio.sleep(0.1)
                            summary, content = await TOOL[tool["name"]](
                                session_buffer, tool["attributes"], tool["content"]
                            )
    
                            # Check for image urls
                            image_urls = extract_image_urls(content)
    
                            if image_urls:
                                figure_summary = await self.query_vision_model(
                                    VISION_MODEL_PROMPT(), image_urls, __event_emitter__
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
    
                    log.debug(f"Current round: {round_count+1}, create_new_round: {create_new_round}")

                    # Reset variables
                    round_buffer.reset()
                    round_count += 1

                log.debug(messages[1:])

            log.info("Chat finished succesfully.")

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
        Initialize knowledge base and store in collection_name_ids dict
        """
        log.debug("Initializing knowledge bases")
        collection_name_ids: Dict[str, str] = {}  # 初始化知识库字典
        try:
            long_chat_collection_name = f"{self.model_id}-Chat-History-{user_id}"
            long_chat_file_name = f"chat-{self.model_id}-{user_id}-{chat_id}.txt"
            long_chat_knowledge = None
            knowledge_bases = Knowledges.get_knowledge_bases_by_user_id(
                user_id, "read"
            )
            rag_collection_names = [
                name.strip() for name in self.valves.RAG_COLLECTION_NAMES.split(",")
            ]
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
            log.debug("Checking last user message for images")
            if messages[-1]["role"] == "user":
                content = messages[-1]["content"]
                if isinstance(content, List):
                    text_content = ""
                    # Find text
                    for c in content:
                        if c.get("type", "") == "text":
                            text_content = c.get("text", "")
                            log.debug(
                                f"Found text in last user message: {text_content}"
                            )
                            break

                    # Find image
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
                                    VISION_MODEL_PROMPT(), [image_url], event_emitter
                                )
                                # insert to message content
                                text_content += vision_summary
                    # Replace
                    messages[-1]["content"] = text_content
                else:
                    image_urls = extract_image_urls(content)
                    if image_urls:
                        log.debug(f"Found image in last user message: {image_urls}")
                        # Call Vision Language Model
                        vision_summary = await self.query_vision_model(
                            VISION_MODEL_PROMPT(), image_urls, event_emitter
                        )
                        messages[-1]["content"] += vision_summary

            # Make user message text-only
            log.debug("Checking all user messages content format")
            for msg in messages:
                if msg["role"] == "user":
                    content = msg["content"]
                    if isinstance(content, List):
                        log.debug("Found a list of content in user message")
                        text_content = ""
                        # Find text content
                        for c in content:
                            if c.get("type", "") == "text":
                                text_content = c.get("content", "")
                                log.debug(f"Found text in user message: {text_content}")
                                break

                        # Replace message
                        log.debug("Replacing user message content")
                        msg["content"] = text_content

    def is_answering(self, text):
        if f"{chr(0x003C)}think{chr(0x003E)}\n\n" in text and f"\n{chr(0x003C)}/think{chr(0x003E)}\n\n" in text:
            return True
        elif f"{chr(0x003C)}think{chr(0x003E)}\n\n" not in text and f"\n{chr(0x003C)}/think{chr(0x003E)}\n\n" not in text:
            return True
        else:
            return False

    def process_content(
        self, delta: dict, round_buffer: RoundBuffer, event_flags: EventFlags
    ) -> str:
        """直接返回处理后的内容"""
        think_open = f"{chr(0x003C)}think{chr(0x003E)}\n\n"
        think_close = f"\n{chr(0x003C)}/think{chr(0x003E)}\n\n"
        if delta.get("reasoning_content", ""):
            content = delta.get("reasoning_content", "")
            if not think_open in round_buffer.total_response:
                content = think_open + content
            round_buffer.total_response += content
            return content
        elif delta.get("content", ""):
            content = delta.get("content", "")
            if (not round_buffer.prefix_mode) and (not think_close in round_buffer.total_response):
                content = think_close + content
                round_buffer.prefix_mode = True
            round_buffer.total_response += content
            return content
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
        prefix_reasoning: bool = True,
    ):
        if not prefix_reasoning:
            log.debug("Removing prefix reasoning")
            pattern = f"{chr(0x003C)}think{chr(0x003E)}\n\n(.*?)\n{chr(0x003C)}/think{chr(0x003E)}\n\n"
            round_buffer.total_response = re.sub(pattern, '', round_buffer.total_response, flags=re.DOTALL)

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

            embeddings = await get_embeddings([window], self.valves.EMBEDDING_MODEL, self.valves.EMBEDDING_API_BASE_URL, self.valves.EMBEDDING_API_KEY)
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

            new_knowledge_names, new_keywords = await generate_query_keywords(query_keywords, collection_name_ids, self.valves.TASK_MODEL, self.valves.TASK_MODEL_API_BASE_URL, self.valves.TASK_MODEL_API_KEY)
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
            query_embeddings = await get_embeddings(query_keywords, self.valves.EMBEDDING_MODEL, self.valves.EMBEDDING_API_BASE_URL, self.valves.EMBEDDING_API_KEY)
            if len(query_embeddings) == 0:
                log.warning(f"Failed to get embedding for queries: {query_keywords}")
                return []

            # Search for each knowledge
            all_results = []
            for knowledge_name in knowledge_names:
                collection_id = collection_name_ids[knowledge_name]
                log.debug(f"Searching collection: {knowledge_name}")
                results = await search_chroma_db(
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
            template_string += DARKSHINE_PROMPT()

        if self.valves.USE_BESIII_GUIDE:
            template_string += BESIII_PROMPT()

        # Task Prompt
        template_string += GUIDE_PROMPT()

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
        work_dir=f"run-{session_buffer.chat_id[:8]}"

        if code_type == "exec":
            # Execute the code
            if filename:
                try:
                    result = session_buffer.code_worker.write_code(
                        file_path=filename,
                        work_dir=work_dir,
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
                    result = session_buffer.code_worker.run_command(command=content, work_dir=work_dir, timeout=300)
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
                    file_path=filename, work_dir=work_dir, content=content
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
                        file_path=filename, original=original, updated=updated, work_dir=work_dir
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
                generate_vl_response(
                    prompt=prompt,
                    image_url=url,
                    model=self.valves.VISION_MODEL,
                    url=self.valves.TASK_MODEL_API_BASE_URL,
                    key=self.valves.TASK_MODEL_API_KEY,
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
