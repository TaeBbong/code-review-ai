from __future__ import annotations
from typing import Sequence

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama

from .base import LLMAdapter


class OllamaAdapter(LLMAdapter):
    def __init__(self, model: str, base_url: str, temperature: float, max_tokens: int):
        self.chat = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens,
            format="json",
        )

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> str:
        res = await self.chat.ainvoke(messages)
        return res.content if hasattr(res, "content") else str(res)
    