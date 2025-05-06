# utils/tools.py
import logging
import json
import os, sys
from typing import AsyncGenerator, Callable, Awaitable, Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
import re
import numpy as np
import asyncio
import threading
from queue import Queue
import httpx
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prompt import (
    DEFAULT_CODE_INTERFACE_PROMPT,
    DEFAULT_WEB_SEARCH_PROMPT,
    DARKSHINE_PROMPT,
    BESIII_PROMPT,
    GUIDE_PROMPT,
    DEFAULT_QUERY_GENERATION_PROMPT,
    VISION_MODEL_PROMPT,
)

log = logging.getLogger(__name__)


@dataclass
class VectorDBResultObject:
    """
    Vector DB search result object
    """
    id_: str
    distance: float
    document: str
    metadata: Dict
    query_embedding: List


class ManagedThread(threading.Thread):
    """
    Managed thread class for automatic removal from manager when thread ends
    """
    def __init__(self, manager, target, args, kwargs):
        super().__init__(target=target, args=args, kwargs=kwargs)
        self.manager = manager  # 持有管理器实例的引用

    def run(self):
        try:
            super().run()  # 执行目标函数
        except Exception as e:
            log.error(f"Thread error: {e}")
        finally:
            self.manager.remove_thread(self)  # 确保无论是否异常都执行移除操作

class ThreadManager:
    """
    Thread manager class
    """
    def __init__(self):
        self.threads = []  # 存储活跃线程的容器
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

async def transfer_userproxy_role(messages):
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

async def merge_adjacent_roles(messages):
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

def extract_json(content):
    # 匹配 ```json 块中的 JSON
    json_block_pattern = r"```json\s*({.*?})\s*```"
    # 匹配 ``` 块中的 JSON
    block_pattern = r"```\s*({.*?})\s*```"
    #log.debug(f"Content: {content}")
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

async def oai_chat_completion(model: str,
                              url: str,
                              api_key: str,
                              body: dict) -> AsyncGenerator[dict, None]:
        data_prefix="data: "
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Initialize client
        client = httpx.AsyncClient(
            http2=True,
            timeout=None,
        )
        payload = {**body, "model": model}
        try:
            async with client.stream(
                "POST",
                f"{url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=None,
            ) as response:
                # 错误处理
                if response.status_code != 200:
                    error = await response.aread()
                    error_str = error.decode(errors="ignore")
                    err_msg = json.loads(error_str).get("message", error_str)[:200]
                    yield {"error": f"{response.status_code}: {err_msg}"}
                    return

                # 流式处理响应
                async for line in response.aiter_lines():
                    if not line.startswith(data_prefix):
                        continue

                    json_str = line[len(data_prefix) :]

                    # 去除首尾空格后检查是否为结束标记
                    if json_str.strip() == "[DONE]":
                        return

                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        error_detail = f"[OAI Chat Completion] Error: failed to extract：{json_str}，Reason：{e}"
                        yield {"error": error_detail}
                        return

                    choice = data.get("choices", [{}])[0]

                    # 结束条件判断
                    if choice.get("finish_reason"):
                        break

                    yield choice
        except Exception as e:
            yield {"error": f"[OAI Chat Completion] Error: {e}"}


def strip_triple_backtick(text: str) -> str:
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
 

async def search_chroma_db(
    collection_id, embeddings: List, top_k: int, max_distance: float
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

async def generate_query_keywords(contexts: List, collection_name_ids, model, base_url, api_key):
    """
    Generate query keywords using llm
    """
    query_template = DEFAULT_QUERY_GENERATION_PROMPT()

    # Create a Jinja2 Template object
    template = Template(query_template)
    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y-%m-%d")

    # Render the template with a list of items
    replace = {
        "CONTEXTS": contexts,
        "COLLECTION_NAMES": list(collection_name_ids.keys())
    }
    query_prompt = template.render(**replace)
    task_messages = [{"role": "user", "content": query_prompt}]
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": task_messages,
    }
    collection_names = []
    keywords = []
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
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

def clean_text(text):
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

async def get_embeddings(texts: List[str], model, base_url, api_key) -> List[List[float]]:
    """调用API获取文本嵌入"""
    embeddings = []
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.post(
            f"{base_url}/embeddings",
            headers=headers,
            json={
                "model": model,
                "input": [clean_text(text) for text in texts],
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

async def generate_vl_response(
    prompt: str,
    image_url: str,
    model: str,
    url: str,
    key: str,
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

