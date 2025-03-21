"""
title: Yui-000
author: Xuliang (xuliangz@sjtu.edu.cn)
description: OpenWebUI pipe function for Yui-000
version: 0.0.0
licence: MIT
"""

import logging
import io, sys
import json
import httpx
import re
import requests
import time
from typing import AsyncGenerator, Callable, Awaitable, Optional, Dict, List, Tuple
from pydantic import BaseModel, Field
import asyncio
from jinja2 import Template
from datetime import datetime
from dataclasses import dataclass
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
from hepai import HRModel
import numpy as np

log = logging.getLogger(__name__)
log.setLevel("DEBUG")

@dataclass
class VectorDBResultObject:
    id_: str
    distance: float
    document: str
    metadata: Dict
    query_embedding: List

class EventFlags:
    def __init__(self):
        self.thinking_state: int = -1
        self.early_end_round = False
        self.mem_updated = False

class SessionBuffer:
    def __init__(self):
        self.rag_queue = []
        # TODO: Memory
        # Sensor
        # 写入文件、读取文件

class RoundBuffer:
    def __init__(self):
        self.sentence_buffer: str = ""
        self.sentences: List[str] = []
        self.window_embeddings: List = []
        self.paragraph_begin_id: int = 0
        self.total_response = ""

    def reset(self):
        self.sentence_buffer = ""
        self.sentences = []
        self.window_embeddings = []
        self.paragraph_begin_id = 0
        self.total_response = ""


class Pipe:
    class Valves(BaseModel):
        # 模型配置
        MODEL_API_BASE_URL: str = Field(
            default="https://aiapi001.ihep.ac.cn/apiv2",
            description="语言模型API的基础请求地址",
        )
        MODEL_API_KEY: str = Field(default="", description="用于身份验证的API密钥")
        BASE_MODEL: str = Field(
            default="deepseek-ai/deepseek-r1:671b",
            description="对话的模型名称",
        )
        EMBEDDING_MODEL: str = Field(
            default="hepai/bge-m3:latest",
            description="用于获取文本嵌入的模型名称",
        )
        # RAG配置
        REALTIME_RAG: bool = Field(default=True, description="Realtime seaching Vector DB")
        RAG_COLLECTION_NAMES: str = Field(default="DarkSHINE_Simulation_Software")
        EMBEDDING_BATCH_SIZE: int = Field(
            default=2000,
            description="Batch size for RAG",
        )
        # 环境交互配置
        REALTIME_IO: bool = Field(default=True, description="Realtime Interact with environment and sense environment change")
        # 提示词配置
        USE_DARKSHINE_GUIDE: bool = Field(default=False, title="Use DarkSHINE Guide")
        USE_BESIII_GUIDE: bool = Field(default=False, title="Use BESIII Guide")
        # 工具配置
        USE_CODE_INTERFACE: bool = Field(default=False)
        USE_MAPPING: bool = Field(default=False)
        USE_WEB_SEARCH: bool = Field(default=False)
        GOOGLE_PSE_API_KEY: str = Field(default="", title="Google PSE API Key")
        GOOGLE_PSE_ENGINE_ID: str = Field(default="", title="Google PSE Engine ID")
        # 其他配置
        MAX_LOOP: int = Field(
            default=20, description="Prevent dead loop, 0 for unlimited."
        )

    def __init__(self):
        # Shared Configs
        self.model_id = "Yui-000"
        self.valves = self.Valves()
        self.data_prefix = "data: "
        self.replace_tags = {"web_search": "Searching"}
        self.rag_queue_max = 1

    def pipes(self):

        return [
            {
                "id": self.model_id,
                "name": self.model_id,
            }
        ]

    async def pipe(
        self, body: dict, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> AsyncGenerator[str, None]:
        # 验证配置
        if not self.valves.MODEL_API_KEY:
            yield json.dumps({"error": "未配置API密钥"}, ensure_ascii=False)
            return
        # 准备请求参数
        headers = {
            "Authorization": f"Bearer {self.valves.MODEL_API_KEY}",
            "Content-Type": "application/json",
        }
        try:
            #==================================================================
            # 初始化变量
            #==================================================================
            event_flags: EventFlags = EventFlags()
            session_buffer: SessionBuffer = SessionBuffer()
            round_buffer: RoundBuffer = RoundBuffer()

            if self.valves.REALTIME_RAG or self.valves.REALTIME_IO:
                do_semantic_segmentation = True

            # Initialize client
            client = httpx.AsyncClient(
                http2=True,
                timeout=None,
            )

            # Initialize tools
            TOOL = {}
            prompt_templates = {}
            code_worker = None
            code_worker_op_system = "Linux"

            if self.valves.USE_CODE_INTERFACE:
                TOOL["code_interface"] = self._code_interface
                prompt_templates["code_interface"] = (
                    self.DEFAULT_CODE_INTERFACE_PROMPT()
                )
                code_worker, code_worker_op_system = self.init_code_worker()
            if self.valves.USE_WEB_SEARCH:
                TOOL["web_search"] = self._web_search
                prompt_templates["web_search"] = self.DEFAULT_WEB_SEARCH_PROMPT()

            # 获取chat id, user id, message id
            user_id, chat_id, message_id = self.extract_event_info(__event_emitter__)

            # 初始化知识库
            knowleges, long_chat_db_name = await self.init_knowledge(user_id)

            # 获取请求参数
            payload = {**body, "model": self.valves.BASE_MODEL}
            messages = payload["messages"]

            #==================================================================
            # 预处理消息（规范化、解析图片）
            #==================================================================
            # TODO
            # self.process_message_figures(messages)
            # User proxy转移到User 角色以保护身份认同
            await self.transfer_userproxy_role(messages)
            # 处理消息以防止相同的角色
            await self.merge_adjacent_roles(messages)
            # TODO
            # self.set_system_prompt(messages)

            log.debug("Old message:")
            log.debug(messages[1:])

            #==================================================================
            # 发起API请求
            #==================================================================
            create_new_round = True
            round_count = 0
            while create_new_round and round_count < self.valves.MAX_LOOP:
                
                # RAG队列大于阈值时等待，再继续下一轮
                if len(session_buffer.rag_queue) > self.rag_queue_max:
                    await asyncio.sleep(0.02)
                    continue

                log.info("Starting chat round")
                async with client.stream(
                    "POST",
                    f"{self.valves.MODEL_API_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=None,
                ) as response:
                    # 错误处理
                    if response.status_code != 200:
                        error = await response.aread()
                        yield self._format_error(response.status_code, error)
                        return

                    # 流式处理响应
                    async for line in response.aiter_lines():
                        if not line.startswith(self.data_prefix):
                            continue
                        #======================================================
                        # 解析数据
                        #======================================================
                        json_str = line[len(self.data_prefix) :]

                        # 去除首尾空格后检查是否为结束标记
                        if json_str.strip() == "[DONE]":
                            return

                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            error_detail = f"解析失败 - 内容：{json_str}，原因：{e}"
                            yield self._format_error("JSONDecodeError", error_detail)
                            return

                        choice = data.get("choices", [{}])[0]
                        
                        #======================================================
                        # 结束条件判断
                        #======================================================
                        if choice.get("finish_reason") or event_flags.early_end_round:
                            log.info("Finishing chat")
                            await self.finalize_sentences(session_buffer, round_buffer)
                            round_buffer.reset()

                            # TODO
                            create_new_round = False
                            break
                        #======================================================
                        # 思考状态处理
                        #======================================================
                        state_output = await self.update_thinking_state(choice.get("delta", {}), event_flags)
                        if state_output:
                            yield state_output  # 直接发送状态标记
                            if state_output == "<think>":
                                await asyncio.sleep(0.02)
                                yield "\n"
                        #======================================================
                        # 内容处理
                        #======================================================
                        content = self.process_content(choice["delta"])
                        if content:
                            yield content
                            if do_semantic_segmentation:
                                # 根据语义分割段落
                                sentence_n = await self.update_sentence_buffer(content, round_buffer)
                                if sentence_n == 0:
                                    continue
                                new_paragraphs = await self.semantic_segmentation(sentence_n, round_buffer)
                                if len(new_paragraphs) == 0:
                                    continue
                                # 实时RAG搜索
                                if self.valves.REALTIME_RAG:
                                    session_buffer.rag_queue.extend(new_paragraphs)

                        # 判断是否打断当前生成并进入下一轮
                        if self.check_break_and_new_round(session_buffer, event_flags):
                            break
                      
                log.debug(messages[-1:])
                round_count += 1

        except Exception as e:
            yield self.format_exception(e)

    def check_break_and_new_round(self, session_buffer: SessionBuffer, event_flags: EventFlags):
        if_break = False
        # 工作记忆更新时打断
        if event_flags.mem_updated:
            break_flag = True
        # RAG队列长度超过阈值时打断
        if len(session_buffer.rag_queue) > self.rag_queue_max:
            break_flag = True
        # Cleanup
        if if_break:
            log.info("Breaking chat round and resetting flags")
            event_flags.mem_updated = False
        return if_break
            

    def extract_event_info(self, event_emitter: Callable[[dict], Awaitable[None]]) -> Tuple[str, str, str]:
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

    def format_exception(self, e: Exception) -> str:
        err_type = type(e).__name__
        return json.dumps({"error": f"{err_type}: {str(e)}"}, ensure_ascii=False)

    async def init_knowledge(self, user_id: str) -> Tuple[Dict, str]:
        """
        初始化知识库数据，将其存储在 knowledges 字典中。
        :param user_id: 用户ID
        :return: 知识库字典和长期记忆库名称
        """
        log.debug("Initializing knowledge bases")
        knowledges: Dict[str, Knowledges] = {}  # 初始化知识库字典
        try:
            long_chat_db_name = f"{self.model_id}-LTM-{user_id}"  # 长期记忆
            knowledge_bases = Knowledges.get_knowledge_bases_by_user_id(
                user_id, "read"
            )  # 获取知识库
            # 获取知识库名称列表
            rag_collection_names = [name.strip() in self.valves.RAG_COLLECTION_NAMES.split(',')]
            # 遍历知识库列表
            for knowledge in knowledge_bases:
                knowledge_name = knowledge.name  # 获取知识库名称
                if knowledge_name in rag_collection_names or knowledge_name == long_chat_db_name:
                    log.info(f"Adding knowledge base: {knowledge_name}")
                    knowledges[knowledge_name] = (
                        knowledge  # 将知识库信息存储到字典中
                    )

            if not long_chat_db_name in knowledges:
                log.info(
                    f"Creating long term memory knowledge base: {long_chat_db_name}"
                )
                form_data = KnowledgeForm(
                    name=long_chat_db_name,
                    description=f"Long term memory for ${self.model_id}",
                    access_control={},
                )
                knowledges[long_chat_db_name] = (
                    Knowledges.insert_new_knowledge(user_id, form_data)
                )

            log.info(
                f"Loaded {len(knowledges)} knowledge bases: {list(knowledges.keys())}"
            )

        except Exception as e:
            raise Exception(f"Failed to initialize knowledge bases: {str(e)}")

        return knowledges, long_chat_db_name

    async def process_message_figures(self):
        pass

    async def transfer_userproxy_role(self, messages):
        log.info("Transferring user proxy messages to user role")
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg["role"] == "assistant":
                # 删除所有running提示
                msg["content"].replace(
                    '<details type="status">\n<summary>Running...</summary>\nRunning\n</details>',
                    "",
                )

                # 用正则匹配所有<details type="user_proxy">内容
                user_proxy_nodes = re.findall(
                    r'<details type="user_proxy">(.*?)</details>',
                    msg["content"],
                    flags=re.DOTALL,
                )

                if user_proxy_nodes:
                    user_contents = []
                    for user_proxy_node in user_proxy_nodes:
                        user_proxy_text = str(user_proxy_node)
                        summary_node = re.search(
                            r"<summary>(.*?)</summary>",
                            user_proxy_text,
                            flags=re.DOTALL,
                        )
                        if summary_node:
                            summary_text = summary_node.group(1).strip()
                        else:
                            summary_text = ""
                        user_proxy_text = re.sub(
                            r"<summary>.*?</summary>",
                            "",
                            user_proxy_text,
                            flags=re.DOTALL,
                        ).strip()
                        user_contents.append(f"{summary_text}\n\n{user_proxy_text}")
                    merged_user_contents = "\n\n".join(user_contents)

                    # (1) 删除消息中的<user_proxy>标签（保留其他内容）
                    clean_content = re.sub(
                        r'<details type="user_proxy">.*?</details>',
                        "",
                        msg["content"],
                        flags=re.DOTALL,
                    ).strip()

                    msg["content"] = clean_content

                    new_user_msg = {"role": "user", "content": merged_user_contents}
                    messages.insert(i + 1, new_user_msg)  # 在当前消息后插入
                    i += 1

            i += 1

    async def merge_adjacent_roles(self, messages):
        log.info("Merging adjacent messages with the same role")
        i = 0
        while i < len(messages) - 1:
            if messages[i]["role"] == messages[i + 1]["role"]:
                # 合并相同角色的消息
                combined_content = (
                    messages[i]["content"] + "\n" + messages[i + 1]["content"]
                )
                messages[i]["content"] = combined_content
                messages.pop(i + 1)
            i += 1

    async def update_thinking_state(self, delta: dict, event_flags: EventFlags) -> str:
        """更新思考状态"""
        state_output = ""

        # 状态转换：未开始 -> 思考中
        if event_flags.thinking_state == -1 and delta.get("reasoning_content"):
            event_flags.thinking_state = 0
            state_output = "<think>"

        # 状态转换：思考中 -> 已回答
        elif (
            event_flags.thinking_state == 0
            and not delta.get("reasoning_content")
            and delta.get("content")
        ):
            event_flags.thinking_state = 1
            state_output = "\n</think>\n\n"

        return state_output

    def process_content(self, delta: dict, round_buffer: RoundBuffer) -> str:
        """直接返回处理后的内容"""
        if delta.get("reasoning_content", ""):
            return delta.get("reasoning_content", "")
        elif delta.get("content", ""):
            delta = delta.get("content", "")
            round_buffer.total_response += delta
            return delta

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
                mark_pattern = r"^[\s\W_]+$"  # 纯符号
                if (
                    re.match(mark_pattern, splits[i].strip())
                    and len(round_buffer.sentences) > 0
                ):
                    round_buffer.sentences[-1] += splits[i]
                else:
                    round_buffer.sentences.append(splits[i])
            round_buffer.sentence_buffer = splits[-1]
            sentence_len_new = len(round_buffer.sentences)
            return sentence_len_new - sentence_len_old
        return 0

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

    async def get_single_embedding(self, text):
        """调用API获取文本嵌入"""
        headers = {"Authorization": f"Bearer {self.valves.MODEL_API_KEY}"}
        response = requests.post(
            f"{self.valves.MODEL_API_BASE_URL}/embeddings",
            headers=headers,
            json={
                "model": self.valves.EMBEDDING_MODEL,
                "input": [self.clean_text(text)],
            },
        )
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            raise Exception(f"API Error: {response.status_code} - {response.text}")

        data = response.json()

        if "data" in data:
            try:
                return data["data"][0]["embedding"]
            except Exception as e:
                raise ValueError(f"Error extracting 'embedding' from response: {e}")
        else:
            raise ValueError("Response from Embedding API did not contain 'data'.")

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

        for i in range(len(round_buffer.sentences) - sentence_n, len(round_buffer.sentences)):
            begin_idx = i - window_size + 1
            if begin_idx < 0:
                continue

            log.debug("Sliding window embedding...")
            window = "".join(round_buffer.sentences[begin_idx : i + 1])

            if window.strip() == "":
                continue

            round_buffer.window_embeddings.append(await self.get_single_embedding(window))
            if len(round_buffer.window_embeddings) > 1:
                similarity = self.cosine_similarity(
                    round_buffer.window_embeddings[-2], round_buffer.window_embeddings[-1]
                )
                if similarity < sim_threshold:
                    paragraph = "".join(
                        round_buffer.sentences[round_buffer.paragraph_begin_id : begin_idx]
                    )
                    new_paragraphs.append(paragraph)
                    round_buffer.paragraph_begin_id = begin_idx
                    log.debug(
                        f"New paragraph:\n {paragraph}\nNext Similarity: {similarity}"
                    )

            return new_paragraphs

    async def finalize_sentences(self, session_buffer: SessionBuffer, round_buffer: RoundBuffer) -> int:
        """
        将缓冲区中的文本添加到句子列表中。
        :param session_buffer: 会话缓冲区
        :param round_buffer: 轮次缓冲区
        :return: 添加到句子列表的句子数量
        """
        sentence_n = 0
        if round_buffer.sentence_buffer:
            sentence_len_old = len(round_buffer.sentences)
            mark_pattern = r"^[\s\W_]+$"  # 纯符号
            if (
                re.match(mark_pattern, round_buffer.sentence_buffer.strip())
                and len(round_buffer.sentences) > 0
            ):
                round_buffer.sentences[-1] += round_buffer.sentence_buffer
            else:
                round_buffer.sentences.append(round_buffer.sentence_buffer)
            round_buffer.sentence_buffer = ""
            sentence_len_new = len(round_buffer.sentences)
            sentence_n =  sentence_len_new - sentence_len_old
        if len(round_buffer.sentences) - round_buffer.paragraph_begin_id > 0:
            paragraph = "".join(
                round_buffer.sentences[round_buffer.paragraph_begin_id:]
            )
            session_buffer.rag_queue.append(paragraph)
            log.debug(
                f"Add rag queue:\n {paragraph}"
            )
            # TODO: submit async rag
            session_buffer.rag_queue.pop()
    
    async def point_search_vector_db_file(file_name, embedding, top_k, max_distance):
        result = VECTOR_DB_CLIENT.search(
            collection_name=file_name,
            vectors=[embedding],
            limit=top_k,
        )
        # sanity check
        if not result:
            return []

        if not all(
            hasattr(result, attr)
            for attr in ["ids", "distances", "documents", "metadatas"]
        ):
            return []

        if (
            not result.ids
            or not result.distances
            or not result.documents
            or not result.metadatas
        ):
            return []

        result_objects = []

        for i in range(len(result.ids)):
            if result.distances[i] > max_distance:
                continue
            result_objects.append(
                VectorDBResultObject(
                    id_=result.ids[i],
                    distance=result.distances[i],
                    document=result.documents[i],
                    metadata=result.metadatas[i],
                    query_embedding=embedding,
                )
            )

        return result_objects

    async def query_collection(
            self, query_keywords: str, knowledges: Dict[str, Knowledges], knowledge_names: List[str] = [], top_k: int = 1, max_distance: float = 0.5
    ) -> list:
        embeddings = None
        query_keywords = query_keyworsd.strip()
        # Generate query embedding
        log.debug(f"Generating Embeddings")
        query_embedding = await self.get_single_embedding(query_keywords)
        # Get knowledge object
        if len(knowledge_names) == 0:
            knowledge_names = knowledges.keys()
        file_names = []
        for knowledge_name in knowledge_names:
            knowledge = knowledges[knowledge_name]
            file_names.extend(["file-" + file_id for file_id in knowledge.data["file_ids"]])
        # Parallel search for files
        log.debug(f"Searching {len(file_names)} files in Knowledges: {knowledge_names}")
        tasks = [point_search_vector_db_file(file_name, query_embedding, top_k, max_distance) for file_name in file_names]
        results = await asyncio.gather(*tasks)
        # flatten
        all_results = [result for sublist in results for result in sublist]
        # sort by distance
        all_results.sort(key=lambda x: x.distance)
        top_results = all_results[:top_k]

        return top_results

    def DEFAULT_CODE_INTERFACE_PROMPT(self):
        return ""

    def DEFAULT_WEB_SEARCH_PROMPT(self):
        return ""

