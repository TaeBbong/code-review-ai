"""
Native OpenAI API adapter for GPT models.

Use this for official OpenAI API (GPT-4o, GPT-4-turbo, etc.)
"""

from __future__ import annotations

from typing import Sequence, Type, TypeVar

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .base import LLMAdapter

T = TypeVar("T", bound=BaseModel)


class OpenAINativeAdapter(LLMAdapter):
    """
    Native OpenAI API adapter for GPT-4o, GPT-4-turbo, etc.

    Uses the official OpenAI API endpoint (https://api.openai.com/v1).
    Requires OPENAI_API_KEY environment variable or explicit api_key.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.chat = ChatOpenAI(
            model=model,
            api_key=api_key,  # None이면 OPENAI_API_KEY 환경변수 사용
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._model = model

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> str:
        res = await self.chat.ainvoke(messages)
        return res.content if hasattr(res, "content") else str(res)

    async def ainvoke_structured(
        self, messages: Sequence[BaseMessage], schema: Type[T]
    ) -> T:
        """
        Structured output using OpenAI's JSON mode.
        GPT-4o and later models support this natively.
        """
        structured_llm = self.chat.with_structured_output(schema)
        return await structured_llm.ainvoke(messages)

    def get_chat_model(self) -> ChatOpenAI:
        """Return ChatOpenAI instance for direct use."""
        return self.chat
