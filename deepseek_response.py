<update_assistant_interface>

```
sys.path.append("/Users/xuliang/third_party")
from assistant_utils.tools import ManagedThread, ThreadManager, extract_json, oai_chat_completion

class Assistant:
    def __init__(self, valves):
        self.valves = valves
        self.data_prefix = "data: "

    async def interface(
        self, body: dict, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> AsyncGenerator[str, None]:
        choices = oai_chat_completion(
            model=self.valves.BASE_MODEL,
            url=self.valves.MODEL_API_BASE_URL,
            api_key=self.valves.MODEL_API_KEY,
            body=body
        )

        async for choice in choices:
            if choice.get("finish_reason"):
                break
            
            if choice.get("error"):
                error_detail = choice.get("error", "")
                yield error_detail
                break

            content = choice.get("delta", {}).get("content", "")
            if content:
                yield content
```

</update_assistant_interface>
