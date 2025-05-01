<update_assistant_interface>
class Assistant:
    def __init__(self, valves):
        self.valves = valves
        self.data_prefix = "data: "
    async def interface(
        self, body: dict, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> AsyncGenerator[str, None]:
        headers = {
            "Authorization": f"Bearer {self.valves.MODEL_API_KEY}",
            "Content-Type": "application/json",
        }
        # Initialize client
        client = httpx.AsyncClient(
            http2=True,
            timeout=None,
        )
        # 获取请求参数
        payload = {**body, "model": self.valves.BASE_MODEL}
        messages = payload["messages"]
        try:
            async with client.stream(
                "POST",
                f"{self.valves.MODEL_API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=None,
            ) as response:
                # 流式处理响应
                async for line in response.aiter_lines():
                    if not line.startswith(self.data_prefix):
                        continue

                    json_str = line[len(self.data_prefix) :]

                    # 去除首尾空格后检查是否为结束标记
                    if json_str.strip() == "[DONE]":
                        return

                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        error_detail = f"解析失败 - 内容：{json_str}，原因：{e}"
                        yield error_detail
                        return

                    choice = data.get("choices", [{}])[0]

                    # 结束条件判断
                    if choice.get("finish_reason"):
                        break

                    content = choice.get("delta", {}).get("content", "")
                    if content:
                        yield content
        except Exception as e:
            error_detail = f"请求失败 - 原因：{e}"
            yield error_detail

</update_assistant_interface>
