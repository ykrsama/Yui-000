"""
title: Yui-000
author: Xuliang (xuliangz@sjtu.edu.cn)
description: OpenWebUI pipe function for Yui-000
version: 0.0.0
licence: MIT
"""

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

class ManagedThread(threading.Thread):
    """自定义线程类，用于在线程结束时自动从管理器移除"""
    def __init__(self, manager, target, args, kwargs):
        super().__init__(target=target, args=args, kwargs=kwargs)
        self.manager = manager  # 持有管理器实例的引用

    def run(self):
        try:
            super().run()       # 执行目标函数
        except Exception as e:
            log.error(f"Thread error: {e}")
        finally:
            self.manager.remove_thread(self)  # 确保无论是否异常都执行移除操作

class ThreadManager:
    """线程管理器类"""
    def __init__(self):
        self.threads = []       # 存储活跃线程的容器
        self.lock = threading.Lock()  # 保证线程安全的锁

    def submit(self, target, args=(), kwargs=None):
        """提交新线程到线程池"""
        if kwargs is None:
            kwargs = {}
            
        # 创建托管线程实例
        thread = ManagedThread(self, target, args, kwargs)
        
        # 使用锁保证线程安全地添加线程
        with self.lock:
            self.threads.append(thread)
        
        thread.start()  # 注意：先添加后启动保证移除操作有效性

    def remove_thread(self, thread):
        """从容器中移除已完成的线程"""
        with self.lock:
            if thread in self.threads:
                self.threads.remove(thread)

    def join_all(self):
        """等待所有线程执行完成"""
        # 复制当前线程列表避免遍历时修改
        with self.lock:
            current_threads = list(self.threads)
            
        for thread in current_threads:
            thread.join()  # 等待每个线程完成

    def active_count(self):
        """获取当前活跃线程数量"""
        with self.lock:
            return len(self.threads)

@dataclass
class VectorDBResultObject:
    id_: str
    distance: float
    document: str
    metadata: Dict
    query_embedding: List

class WorkingMemory:
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
        with open(self.filename, 'w') as file:
            json.dump([asdict(obj) for obj in self.objects], file)

    def load(self):
        log.info("Loading working memory...")
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as file:
                data = json.load(file)
                self.objects = [VectorDBResultObject(**obj) for obj in data]
        else:
            log.info("No working memory file found.")

    def __str__(self):
        output = ""
        for obj in self.objects:
            log.debug(f"Result Document: {obj.document}\nDistance: {obj.distance}\nMetadata: {obj.metadata}")
            source = obj.metadata.get("source", "Unknown").replace("~", "/")
            output += f"<context source=\"{source}\">\n{obj.document}\n</context>\n\n"
        return output

    def __repr__(self):
        return str(self.objects)

class EventFlags:
    def __init__(self):
        self.thinking_state: int = 0
        self.early_end_round = False
        self.mem_updated = False

class SessionBuffer:
    def __init__(self, chat_id):
        self.rag_thread_mgr: ThreadManager = ThreadManager()
        self.rag_result_queue: Queue = Queue()
        self.memory: WorkingMemory = WorkingMemory(chat_id)
        # TODO: Sensor

class RoundBuffer:
    def __init__(self):
        self.sentence_buffer: str = ""
        self.sentences: List[str] = []
        self.window_embeddings: List = []
        self.paragraph_begin_id: int = 0
        self.total_response = ""
        self.reasoning_content = ""

    def reset(self):
        self.sentence_buffer = ""
        self.sentences = []
        self.window_embeddings = []
        self.paragraph_begin_id = 0
        self.total_response = ""
        self.reasoning_content = ""

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
        TASK_MODEL: str = Field(
            default="deepseek-ai/deepseek-v3:671b",
            description="用于提取搜索提示词等的模型名称",
        )
        EMBEDDING_API_BASE_URL: str = Field(
            default="http://127.0.0.1:11434/v1",
            description="文本嵌入API的基础请求地址",
        )
        EMBEDDING_API_KEY: str = Field(default="", description="用于身份验证的API密钥")
        EMBEDDING_MODEL: str = Field(
            default="bge-m3:latest",
            description="用于获取文本嵌入的模型名称",
        )
        # RAG配置
        REALTIME_RAG: bool = Field(default=True, description="Realtime seaching Vector DB")
        GENERATE_KEYWORDS_FROM_MODEL: bool = Field(default=True, description="Generate keywords from model")
        RAG_COLLECTION_NAMES: str = Field(default="Yui-000-Source, DarkSHINE_Simulation_Software")
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
        self.rag_thread_max = 1

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
            # 获取chat id, user id, message id
            user_id, chat_id, message_id = self.extract_event_info(__event_emitter__)

            event_flags: EventFlags = EventFlags()
            session_buffer: SessionBuffer = SessionBuffer(chat_id)
            round_buffer: RoundBuffer = RoundBuffer()

            # 初始化知识库
            knowledges, long_chat_db_name = await self.init_knowledge(user_id)

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
            # 更新系统提示词
            self.set_system_prompt(messages, session_buffer, code_worker_op_system)
            # RAG用户消息
            if messages[-1]["role"] == "user":
                results = await self.query_collection(messages[-1]["content"], knowledges)
                for result in results:
                    session_buffer.memory.add_object(result)

            log.debug("Old message:")
            log.debug(messages[1:])

            #==================================================================
            # 发起API请求
            #==================================================================
            create_new_round = True
            round_count = 0
            while create_new_round and round_count < self.valves.MAX_LOOP:
                
                # RAG队列大于阈值时等待，再继续下一轮
                if session_buffer.rag_thread_mgr.active_count() > self.rag_thread_max:
                    await asyncio.sleep(0.1)
                    continue

                # 更新系统提示词
                self.set_system_prompt(messages, session_buffer, code_worker_op_system)

                log.debug(messages[1:])
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
                        # 提前结束条件判断
                        #======================================================
                        # 检查rag结果队列
                        while not session_buffer.rag_result_queue.empty():
                            result = session_buffer.rag_result_queue.get()
                            for result_object in result:
                                session_buffer.memory.add_object(result_object)
                            # TODO 更新工作记忆
                            event_flags.mem_updated = True

                        # 判断是否打断当前生成并进入下一轮
                        if self.check_break_and_new_round(session_buffer, event_flags):
                            break
 
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
                            self.update_assistant_message(messages, round_buffer, event_flags, prefix_reasoning=False)
                            paragraph = await self.finalize_sentences(session_buffer, round_buffer)
                            if paragraph:
                                session_buffer.rag_thread_mgr.submit(
                                    self.query_collection_to_queue,
                                    args=(session_buffer.rag_result_queue, [paragraph], knowledges)
                                )
                            round_buffer.reset()

                            # TODO
                            create_new_round = False
                            round_count += 1
                            break
                        #======================================================
                        # 思考状态处理
                        #======================================================
                        state_output = await self.update_thinking_state(choice.get("delta", {}), event_flags)
                        if state_output:
                            yield state_output  # 直接发送状态标记

                        do_semantic_segmentation = False
                        if self.valves.REALTIME_RAG:
                            do_semantic_segmentation = True
                        if self.valves.REALTIME_IO and self.is_answering(event_flags.thinking_state):
                            do_semantic_segmentation = True
                        #======================================================
                        # 内容处理
                        #======================================================
                        content = self.process_content(choice["delta"], round_buffer, event_flags)
                        if content:
                            yield content
                            self.update_assistant_message(messages, round_buffer, event_flags, prefix_reasoning=True)

                            if do_semantic_segmentation:
                                # 根据语义分割段落
                                sentence_n = await self.update_sentence_buffer(content, round_buffer)
                                if sentence_n == 0:
                                    continue
                                new_paragraphs = await self.semantic_segmentation(sentence_n, round_buffer)
                                if len(new_paragraphs) == 0:
                                    continue
                                # 实时RAG搜索
                                for paragraph in new_paragraphs:
                                    session_buffer.rag_thread_mgr.submit(
                                        self.query_collection_to_queue,
                                        args=(session_buffer.rag_result_queue, [paragraph], knowledges)
                                    )
                     
                log.debug(messages[1:])
        except Exception as e:
            yield self._format_error("Exception", str(e))

    def _format_error(self, status_code: int, error: bytes) -> str:
        # 如果 error 已经是字符串，则无需 decode
        if isinstance(error, str):
            error_str = error
        else:
            error_str = error.decode(errors="ignore")

        try:
            err_msg = json.loads(error_str).get("message", error_str)[:200]
        except Exception as e:
            err_msg = error_str[:200]
        return json.dumps(
            {"error": f"HTTP {status_code}: {err_msg}"}, ensure_ascii=False
        )

    def check_break_and_new_round(self, session_buffer: SessionBuffer, event_flags: EventFlags):
        if_break = False
        # 工作记忆更新时打断
        if event_flags.mem_updated:
            log.info("Breaking chat round: Memory updated")
            if_break = True
            event_flags.mem_updated = False
        # RAG队列长度超过阈值时打断
        if session_buffer.rag_thread_mgr.active_count() > self.rag_thread_max:
            log.info(f"Breaking chat round: RAG thread count {session_buffer.rag_thread_mgr.active_count()}")
            if_break = True

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
            rag_collection_names = [name.strip() for name in self.valves.RAG_COLLECTION_NAMES.split(',')]
            # 遍历知识库列表
            for knowledge in knowledge_bases:
                knowledge_name = knowledge.name  # 获取知识库名称
                if knowledge_name in rag_collection_names or knowledge_name == long_chat_db_name:
                    log.info(f"Adding knowledge base: {knowledge_name} ({knowledge.id})")
                    knowledges[knowledge_name] = (
                        knowledge  # 将知识库信息存储到字典中
                    )

            if not long_chat_db_name in knowledges:
                log.info(
                    f"Creating long term memory knowledge base: {long_chat_db_name}"
                )
                form_data = KnowledgeForm(
                    name=long_chat_db_name,
                    description=f"Long term memory for {self.model_id}",
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

        #log.debug(f"Thinking state old: {event_flags.thinking_state}, delta: {delta}")

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

        #log.debug(f"Thinking state new: {event_flags.thinking_state}")

        return state_output

    def process_content(self, delta: dict, round_buffer: RoundBuffer, event_flags: EventFlags) -> str:
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

    def update_assistant_message(self, messages, round_buffer, event_flags: EventFlags, prefix_reasoning: bool=False):
        """更新助手消息"""
        if prefix_reasoning:
            if round_buffer.total_response:
                assistante_message = {
                    "role": "assistant",
                    "content": f"<think>\n\n{round_buffer.reasoning_content}\n</think>\n\n{round_buffer.total_response}",
                    "prefix": True
                }
            else:
                assistante_message = {
                    "role": "assistant",
                    "content": f"<think>\n\n{round_buffer.reasoning_content}",
                    "prefix": True
                }
                event_flags.thinking_state = 2
        else:
            if not round_buffer.total_response:
                log.error("No total_response, cannot update assistant message?")
                return

            assistante_message = {
                "role": "assistant",
                "content": round_buffer.total_response,
                "prefix": True
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

        for i in range(len(round_buffer.sentences) - sentence_n, len(round_buffer.sentences)):
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

    async def finalize_sentences(self, session_buffer: SessionBuffer, round_buffer: RoundBuffer) -> str:
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
                re.match(mark_pattern, round_buffer.sentence_buffer.strip())   # 纯符号
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
            return paragraph
        return ""
    
    async def search_chroma_db(self, collection_id, embeddings: List, top_k: int, max_distance: float):
        """
        Search Chroma DB
        """
        result_objects = []
        try:
            if not collection_id in VECTOR_DB_CLIENT.client.list_collections():
                log.warning(f"Collection {collection_id} not found in Vector DB, maybe this is an empty collection.")
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
                    if (max_distance > 0) and (results.distances[query_i][k] > max_distance):
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
            self, query_keywords: List[str], knowledges: Dict[str, Knowledges], knowledge_names: List[str] = [], top_k: int = 1, max_distance: float = 0 ) -> list:
        log.debug("Querying Knowledge Collection")
        embeddings = []
        if self.valves.GENERATE_KEYWORDS_FROM_MODEL:
            query_keywords = await self.generate_query_keywords(query_keywords)
        query_keywords = [kwd.strip() for kwd in query_keywords]
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
            # Get knowledge object
            if len(knowledge_names) == 0:
                knowledge_names = knowledges.keys()

            # Search for each knowledge
            all_results = []
            for knowledge_name in knowledge_names:
                knowledge = knowledges[knowledge_name]
                if not knowledge or not knowledge.data:
                    continue
                collection_id = knowledge.id
                log.debug(f"Searching collection: {knowledge_name}")
                results = await self.search_chroma_db(collection_id, query_embeddings, top_k, max_distance)
                if not results:
                    continue
                all_results.extend(results)

            log.debug(f"Search done with {len(all_results)} results")

            return all_results

        except Exception as e:
            log.error(f"Error querying collection: {e}")

        return []

    def query_collection_to_queue(self, result_queue: Queue, query_keywords: List[str], knowledges: Dict[str, Knowledges], knowledge_names: List[str] = [], top_k: int = 1, max_distance: float = 0.0):
        try:
            results = asyncio.run(self.query_collection(query_keywords, knowledges, knowledge_names, top_k, max_distance))
            if results:
                log.debug(f"Put into results")
                result_queue.put(results)
        except Exception as e:
            log.error(f"Error querying collection to queue: {e}")

    def extract_json(self, content):
        # 匹配 ```json 块中的 JSON
        json_block_pattern = r'```json\s*({.*?})\s*```'
        # 匹配 ``` 块中的 JSON
        block_pattern = r'```\s*({.*?})\s*```'
        log.debug(f"Content: {content}")
        try:
            # 尝试匹配 ```json 块
            match = re.search(json_block_pattern, content, re.DOTALL)
            if match:
                return json.loads(match.group(1))
    
            # 尝试匹配 ``` 块
            match = re.search(block_pattern, content, re.DOTALL)
            if match:
                return json.loads(match.group(1))
    
            # 尝试直接转换
            return json.loads(content)
        except Exception as e:
            log.error(f"Failed to extract JSON: {e}")
    
        return None

    async def generate_query_keywords(self, contexts: List):
        """
        Generate query keywords using llm
        """
        query_template = self.DEFAULT_QUERY_GENERATION_PROMPT_TEMPLATE()

        # Create a Jinja2 Template object
        template = Template(query_template)
        current_date = datetime.now()
        formatted_date = current_date.strftime("%Y-%m-%d")
        #if len(messages) > 7:
        #    short_messages = [messages[0]] + messages[-6:]
        #else:
        #    short_messages = messages
        
        # Render the template with a list of items
        replace = {"CURRENT_DATE": formatted_date, "CONTEXTS": contexts}
        query_prompt = template.render(**replace)
        task_messages = [{"role": "user", "content": query_prompt}]
        headers = {"Authorization": f"Bearer {self.valves.MODEL_API_KEY}"}
        payload = {
            "model":  self.valves.TASK_MODEL,
            "messages": task_messages,
        }
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
            result_json = self.extract_json(content)
            keywords = result_json.get("queries", [])
        except Exception as e:
            log.error(f"Failed to generate query keywords: {e}")
        log.debug(f"Generated query keywords: {keywords}")
        return keywords

    def set_system_prompt(self, messages, session_buffer: SessionBuffer, op_system: str):
        # Working memory
        wm_template = "## Context\n\n{{WORKING_MEMORY}}---\n"

        # Create a Jinja2 Template object
        template = Template(wm_template)
        current_date = datetime.now()
        formatted_date = current_date.strftime("%Y-%m-%d")

        # Render the template with a list of items
        context = {"CURRENT_DATE": formatted_date, "OP_SYSTEM": op_system, "WORKING_MEMORY": str(session_buffer.memory)}
        result = template.render(**context)

        # Set system_prompt
        if messages[0]["role"] == "system":
            messages[0]["content"] = result
        else:
            context_message = {"role": "system", "content": result}
            messages.insert(0, context_message)

    def DEFAULT_CODE_INTERFACE_PROMPT(self):
        return ""

    def DEFAULT_WEB_SEARCH_PROMPT(self):
        return ""

    def DEFAULT_QUERY_GENERATION_PROMPT_TEMPLATE(self):
        return """### Task:
Analyze the context to determine the necessity of generating search queries, in the given language. By default, **prioritize generating 1-3 broad and relevant search queries** unless it is absolutely certain that no additional information is required. The aim is to retrieve comprehensive, updated, and valuable information even with minimal uncertainty. If no search is unequivocally needed, return an empty list.

### Guidelines:
- Respond **EXCLUSIVELY** with a JSON object. Any form of extra commentary, explanation, or additional text is strictly prohibited.
- When generating search queries, respond in the format: { "queries": ["query1", "query2"] }, ensuring each query is distinct, concise, and relevant to the topic.
- If and only if it is entirely certain that no useful results can be retrieved by a search, return: { "queries": [] }.
- Err on the side of suggesting search queries if there is **any chance** they might provide useful or updated information.
- Be concise and focused on composing high-quality search queries, avoiding unnecessary elaboration, commentary, or assumptions.
- Today's date is: {{CURRENT_DATE}}.
- Always prioritize providing actionable and broad queries that maximize informational coverage.

### Output:
Strictly return in JSON format: 
{
  "queries": ["query1", "query2"]
}

### Contexts
{% for item in CONTEXTS %}
{{ item }}
{% endfor %}
"""
