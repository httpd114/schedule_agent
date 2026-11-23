"""智谱 GLM OpenAI 兼容客户端，封装 Function Calling 请求/响应。"""
from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from config import settings


@lru_cache(maxsize=1)
def get_llm() -> "ZhipuClient":
    return ZhipuClient()


class ZhipuClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.zhipu_api_key,
            base_url=settings.zhipu_base_url,
            timeout=settings.request_timeout,
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """发起一次 chat completion，统一返回 OpenAI 风格 assistant message。"""
        kwargs: dict = {
            "model": settings.zhipu_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 2048,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        tool_calls = []
        for call in message.tool_calls or []:
            tool_calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            )
        return {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": tool_calls,
        }
