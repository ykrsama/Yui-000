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
from datetime import datetime

log = logging.getLogger(__name__)
log.setLevel("DEBUG")


class Pipe:
    class Valves(BaseModel):
        """
        Configuration for the pipe
        """
        MODEL_API_BASE_URL: str = Field(
            default="https://aiapi001.ihep.ac.cn/apiv2",
            description="语言模型API的基础请求地址",
        )
        MODEL_API_KEY: str = Field(
            default="api key here", description="用于身份验证的API密钥"
        )
        BASE_MODEL: str = Field(
            default="deepseek-ai/deepseek-r1:671b",
            description="对话的模型名称",
        )
        TASK_MODEL: str = Field(
            default="deepseek-ai/deepseek-v3:671b",
            description="用于提取搜索提示词等的模型名称",
        )
        VISION_MODEL: str = Field(
            default="ark/doubao-vision-pro",
            description="用语解析图片内容",
        )
        EMBEDDING_API_BASE_URL: str = Field(
            default="http://127.0.0.1:11434/v1",
            description="文本嵌入API的基础请求地址",
        )
        EMBEDDING_API_KEY: str = Field(
            default="api key here", description="用于身份验证的API密钥"
        )
        EMBEDDING_MODEL: str = Field(
            default="bge-m3:latest",
            description="用于获取文本嵌入的模型名称",
        )
        # RAG配置
        REALTIME_RAG: bool = Field(
            default=True, description="Realtime seaching Vector DB"
        )
        GENERATE_KEYWORDS_FROM_MODEL: bool = Field(
            default=True, description="Generate keywords from model"
        )
        RAG_COLLECTION_NAMES: str = Field(default="Yui-000-Source, Open WebUI Backend")
        EMBEDDING_BATCH_SIZE: int = Field(
            default=2000,
            description="Batch size for RAG",
        )
        # 环境交互配置
        REALTIME_IO: bool = Field(
            default=True,
            description="Realtime Interact with environment and sense environment change",
        )
        # 提示词配置
        USE_DARKSHINE_GUIDE: bool = Field(default=True, title="Use DarkSHINE Guide")
        USE_BESIII_GUIDE: bool = Field(default=False, title="Use BESIII Guide")
        # 工具配置
        USE_CODE_INTERFACE: bool = Field(default=True)
        USE_WEB_SEARCH: bool = Field(default=True)
        USE_MAPPING: bool = Field(default=False)
        CODE_WORKER_NAME: str = Field(
            default="xuliang/code-worker-v2",
            description="Code worker model name",
        )
        CODE_WORKER_BASE_URL: str = Field(
            default="http://localhost:42899/apiv2",
            description="Code worker API base URL",
        )
        GOOGLE_PSE_API_KEY: str = Field(
            default="api key here", title="Google PSE API Key"
        )
        GOOGLE_PSE_ENGINE_ID: str = Field(
            default="id here", title="Google PSE Engine ID"
        )
        # 其他配置
        MAX_LOOP: int = Field(
            default=20, description="Prevent dead loop, 0 for unlimited."
        )

    def __init__(self):
        # Shared Configs
        self.model_id = "Yui-000"
        self.valves = self.Valves()
        self.data_prefix = "data: "

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
            messages = body["messages"]
            assistant_code_block = ""

            log.info("Extracting assistant code block...")
            for i in range(len(messages) - 1, -1, -1):
                content = messages[i]["content"]
                content_text = content
                if isinstance(content, List):
                    for c in content:
                        if c.get("type", "") == "text":
                            content_text = c.get("text", "")
                            break

                code_blocks = re.findall(r'<update_assistant_core>(.*?)</update_assistant_core>', content_text, re.DOTALL)
                if code_blocks:
                    assistant_code_block = self.strip_triple_backtick(code_blocks[-1])
                    break
            if not assistant_code_block:
                log.warning("Fallback to load from default_interface.py")
                default_interface_path = os.path.expanduser("~/third_party/assistant_utils/default_interface.py")
                if os.path.exists(default_interface_path):
                    with open(default_interface_path, 'r') as file:
                        assistant_code_block = file.read()
                else:
                    yield "\nError: Default interface file not found"
                    return
    
            log.info("Evaluating assistant code block")
            namespace = globals().copy()
            # Execute the extracted code within the local_vars context
            exec(assistant_code_block, namespace)
            if "Assistant" not in namespace:
                yield "\nError: Assistant class not found in the assistant interface code"
                return

            Assistant = namespace['Assistant']
            agent = Assistant(self.valves)
            if not agent:
                yield "Error: Failed to create Assistant instance"
                return

            if not callable(agent.interface):
                yield "\nError: No callable function interface() found in the assistant source"
                return

            # Call the interface function with the provided arguments
            result = agent.interface(body, __event_emitter__)
            if isinstance(result, AsyncGenerator):
                async for item in result:
                    yield item
            else:
                yield result

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
            {"error": f"{status_code}: {err_msg}"}, ensure_ascii=False
        )

    def strip_triple_backtick(self, text: str) -> str:
        """
        Strips triple backticks from the text.
        """
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            # Remove the first line and the last line (markdown code block)
            lines = text.splitlines()
            if len(lines) > 1:
                lines = lines[1:-1]
            text = "\n".join(lines)
        return text

