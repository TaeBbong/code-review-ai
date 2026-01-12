from __future__ import annotations
from typing import Sequence, TypeVar

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from .base import LLMAdapter

T = TypeVar("T", bound=BaseModel)


class OllamaAdapter(LLMAdapter):
    def __init__(self, model: str, base_url: str, temperature: float, max_tokens: int):
        self.chat = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens,
            format="json",
        )
        self._model = model
        self._base_url = base_url

    @property
    def provider(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> str:
        res = await self.chat.ainvoke(messages)
        return res.content if hasattr(res, "content") else str(res)

    async def ainvoke_structured(
        self, messages: Sequence[BaseMessage], schema: type[T]
    ) -> T:
        """Ollama도 with_structured_output 지원."""
        structured_llm = self.chat.with_structured_output(schema)
        return await structured_llm.ainvoke(messages)
    