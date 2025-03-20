"""
title: Yui-001
author: Xuliang (xuliangz@sjtu.edu.cn)
description: OpenWebUI pipe function for Yui-001
version: 0.0.1
licence: MIT
"""

import logging
import io, sys
import json
import httpx
import re
import requests
import time
from typing import AsyncGenerator, Callable, Awaitable, Optional, List, Tuple
from pydantic import BaseModel, Field
import asyncio
from jinja2 import Template
from datetime import datetime
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
from open_webui.retrieval.utils import (
    generate_embeddings,
)
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


class RAGResultObject:
    def __init__(self, id, distance, document, metadata):
        self.id = id
        self.distance = distance
        self.document = document
        self.metadata = metadata


class Pipe:
    class Valves(BaseModel):
        # 模型配置
        MODEL_API_BASE_URL: str = Field(
            default="https://aiapi001.ihep.ac.cn/apiv2",
            description="语言模型API的基础请求地址",
        )
        MODEL_API_KEY: str = Field(default="", description="用于身份验证的API密钥")
        MAIN_MODEL: str = Field(
            default="deepseek-ai/deepseek-r1:671b",
            description="对话的模型名称",
        )
        EMBEDDING_MODEL: str = Field(
            default="hepai/bge-m3:latest",
            description="用于获取文本嵌入的模型名称",
        )
        # RAG配置
        RAG_AT_TEST_TIME: bool = Field(default=True)
        RAG_COLLECTION_NAMES: str = Field(default="DarkSHINE_Simulation_Software")
        EMBEDDING_BATCH_SIZE: int = Field(
            default=2000,
            description="Batch size for RAG",
        )
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
        # Configs
        self.valves = self.Valves()
        self.data_prefix = "data: "
        self.client = None
        self.TOOL = {}
        self.prompt_templates = {}
        self.replace_tags = {"web_search": "Searching"}
        # Global vars
        self.user_id: str = ""
        self.chat_id: str = ""
        self.message_id: str = ""
        self.long_term_memory: str = ""
        self.total_response = ""
        self.temp_content = ""  # Temporary string to hold accumulated content
        self.current_tag_name = None
        self.immediate_stop = False
        self.code_worker = None
        self.op_system = "Linux"  # code worker system
        self.rag_queue = []
        self.rag_queue_max = 1
        self.sentence_buffer: str = ""
        self.sentences: List[str] = []
        self.window_embeddings: List[str] = []
        self.paragraph_begin_id = 0

    def pipes(self):
        self.max_loop = self.valves.MAX_LOOP
        if self.valves.USE_CODE_INTERFACE:
            self.TOOL["code_interface"] = self._code_interface
            self.prompt_templates["code_interface"] = (
                self.DEFAULT_CODE_INTERFACE_PROMPT()
            )
            self.init_code_worker()
        else:
            if "code_interface" in self.TOOL.keys():
                self.TOOL.pop("code_interface")
            if "code_interface" in self.prompt_templates.keys():
                self.prompt_templates.pop("code_interface")
        if self.valves.USE_WEB_SEARCH:
            self.TOOL["web_search"] = self._web_search
            self.prompt_templates["web_search"] = self.DEFAULT_WEB_SEARCH_PROMPT()
        else:
            if "web_search" in self.TOOL.keys():
                self.TOOL.pop("web_search")
            if "web_search" in self.prompt_templates.keys():
                self.prompt_templates.pop("web_search")

        self.client = httpx.AsyncClient(
            http2=True,
            timeout=None,
        )

        return [
            {
                "id": "Yui-001",
                "name": "Yui-001",
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
            # 获取chat id, user id, message id
            self.extract_event_info(__event_emitter__)
            # 初始化知识库
            self.init_knowledge()
            # 初始化buffer
            self.sentence_buffer = ""
            self.sentences = []
            self.window_embeddings = []
            self.rag_queue = []
            self.paragraph_begin_id = 0
            # 获取请求参数
            payload = {**body, "model": self.valves.MAIN_MODEL}
            messages = payload["messages"]
            # TODO
            # self.process_message_figures(messages)
            # User proxy转移到User 角色以保护身份认同
            self.transfer_userproxy_role(messages)
            # 处理消息以防止相同的角色
            self.merge_adjacent_roles(messages)
            # TODO
            # self.set_system_prompt(messages)

            log.debug("Old message:")
            log.debug(messages[1:])

            # 发起API请求
            create_new_round = True
            round_count = 0
            while create_new_round and round_count < self.max_loop:
                if len(self.rag_queue) > self.rag_queue_max:
                    await asyncio.sleep(0.02)
                    continue

                thinking_state = {"thinking": -1}  # 使用字典来存储thinking状态
                async with self.client.stream(
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

                        # 截取 JSON 字符串
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

                        # 结束条件判断
                        if choice.get("finish_reason") or self.immediate_stop:
                            log.info("Finishing chat")
                            sentence_n = self.finalize_sentences()
                            if sentence_n:
                                self.process_deltas(sentence_n)

                            # TODO
                            create_new_round = False
                            break

                        # 状态机处理
                        state_output = await self.update_thinking_state(
                            choice.get("delta", {}), thinking_state
                        )
                        if state_output:
                            yield state_output  # 直接发送状态标记
                            if state_output == "<think>":
                                await asyncio.sleep(0.02)
                                yield "\n"

                        # 内容处理
                        content = self.process_content(choice["delta"])
                        if content:
                            # 根据语义分割段落
                            sentence_n = self.update_sentence_buffer(content)
                            if sentence_n:
                                self.process_deltas(sentence_n)

                            yield content

                log.debug(messages[-1:])
                round_count += 1

        except Exception as e:
            yield self.format_exception(e)

    def extract_event_info(self, event_emitter):
        if not event_emitter or not event_emitter.__closure__:
            return
        for cell in event_emitter.__closure__:
            if isinstance(request_info := cell.cell_contents, dict):
                self.user_id = request_info.get("user_id")
                self.chat_id = request_info.get("chat_id")
                self.message_id = request_info.get("message_id")
        return

    def format_exception(self, e: Exception) -> str:
        err_type = type(e).__name__
        return json.dumps({"error": f"{err_type}: {str(e)}"}, ensure_ascii=False)

    def init_knowledge(self):
        """
        初始化知识库数据，将其存储在 self.knowledges 字典中。
        """
        log.debug("Initializing knowledge bases")
        self.knowledges = {}  # 初始化知识库字典
        try:
            self.long_term_memory = "Yui-001-LTM-" + self.user_id  # 长期记忆
            knowledge_bases = Knowledges.get_knowledge_bases_by_user_id(
                self.user_id, "read"
            )  # 获取知识库

            # 遍历知识库列表
            for knowledge in knowledge_bases:
                knowledge_name = knowledge.name  # 获取知识库名称
                if knowledge_name:  # 确保知识库名称存在
                    log.debug(f"Adding knowledge base: {knowledge_name}")
                    self.knowledges[knowledge_name] = (
                        knowledge  # 将知识库信息存储到字典中
                    )
                else:
                    log.warning("Found a knowledge base without a name, skipping it.")

            if not self.long_term_memory in self.knowledges:
                log.info(
                    f"Creating long term memory knowledge base: {self.long_term_memory}"
                )
                form_data = KnowledgeForm(
                    name=self.long_term_memory,
                    description="Long term memory for Yui-001",
                    access_control={},
                )
                self.knowledges[self.long_term_memory] = (
                    Knowledges.insert_new_knowledge(self.user_id, form_data)
                )

            log.info(
                f"Loaded {len(self.knowledges)} knowledge bases: {list(self.knowledges.keys())}"
            )

        except Exception as e:
            raise Exception(f"Failed to initialize knowledge bases: {str(e)}")

    def process_message_figures(self):
        pass

    def transfer_userproxy_role(self, messages):
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

    def merge_adjacent_roles(self, messages):
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

    async def update_thinking_state(self, delta: dict, thinking_state: dict) -> str:
        """更新思考状态机（简化版）"""
        state_output = ""

        # 状态转换：未开始 -> 思考中
        if thinking_state["thinking"] == -1 and delta.get("reasoning_content"):
            thinking_state["thinking"] = 0
            state_output = "<think>"

        # 状态转换：思考中 -> 已回答
        elif (
            thinking_state["thinking"] == 0
            and not delta.get("reasoning_content")
            and delta.get("content")
        ):
            thinking_state["thinking"] = 1
            state_output = "\n</think>\n\n"

        return state_output

    def process_content(self, delta: dict) -> str:
        """直接返回处理后的内容"""
        if delta.get("reasoning_content", ""):
            return delta.get("reasoning_content", "")
        elif delta.get("content", ""):
            delta = delta.get("content", "")
            self.total_response += delta
            return delta

    def update_sentence_buffer(self, text) -> int:
        """
        更新句子缓冲区，将文本分割成句子并添加到句子列表中。
        :param text: 要添加到缓冲区的文本
        :return: 添加到句子列表的句子数量
        """
        self.sentence_buffer += text
        pattern = r'(?<=[。！？“”])|(?<=\n)|(?<=[.!?]\s)|(?<=")'
        splits = re.split(pattern, self.sentence_buffer)
        if len(splits) > 1:
            for i in range(len(splits) - 1):
                mark_pattern = r"^[\s\W_]+$"  # 纯符号
                if (
                    re.match(mark_pattern, splits[i].strip())
                    and len(self.sentences) > 0
                ):
                    self.sentences[-1] += splits[i]
                else:
                    self.sentences.append(splits[i])
            self.sentence_buffer = splits[-1]
            return len(splits) - 1
        return 0

    def finalize_sentences(self) -> int:
        """
        将缓冲区中的文本添加到句子列表中。
        :return: 添加到句子列表的句子数量
        """
        if self.sentence_buffer:
            self.sentences.append(self.sentence_buffer)
            self.sentence_buffer = ""
            return 1
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

    def get_embeddings(self, text):
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

    def process_deltas(self, sentence_n):
        # 滑动窗口获取嵌入向量
        window_size = 3
        window_step = 1
        sim_threshold = 0.8

        for i in range(len(self.sentences) - sentence_n, len(self.sentences)):
            begin_idx = i - window_size + 1
            if begin_idx < 0:
                continue

            log.debug("获取滑动窗口嵌入向量...")
            window = "".join(self.sentences[begin_idx : i + 1])
            log.debug(f"Window: {window}")
            self.window_embeddings.append(self.get_embeddings(window))
            if len(self.window_embeddings) > 1:
                similarity = self.cosine_similarity(
                    self.window_embeddings[-2], self.window_embeddings[-1]
                )
                if similarity < sim_threshold:
                    paragraph = "".join(
                        self.sentences[self.paragraph_begin_id : begin_idx]
                    )
                    self.rag_queue.append(paragraph)
                    self.paragraph_begin_id = begin_idx
                    log.debug(
                        f"Add rag queue: {paragraph}\nNext Similarity: {similarity}"
                    )
                    # TODO: submit async rag
                    self.rag_queue.pop()

    def DEFAULT_CODE_INTERFACE_PROMPT(self):
        return ""

    def DEFAULT_WEB_SEARCH_PROMPT(self):
        return ""

