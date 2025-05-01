# utils/tools.py
import logging
import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import re
import numpy as np
import asyncio
import threading
from queue import Queue

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

