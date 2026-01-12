from __future__ import annotations
from typing import Sequence, Type, TypeVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from .base import LLMAdapter

from langchain_openai import ChatOpenAI

T = TypeVar("T", bound=BaseModel)


class OpenAICompatAdapter(LLMAdapter):
    """
    vLLM이 OpenAI-compatible 서버를 띄운 경우(`/v1/chat/completions`),
    ChatOpenAI를 base_url로 붙여서 사용.
    """
    def __init__(self, model: str, base_url: str, api_key: str, temperature: float, max_tokens: int):
        self.chat = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._model = model
        self._base_url = base_url

    @property
    def provider(self) -> str:
        return "vllm"

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
        Structured output을 사용하여 스키마에 맞는 결과 반환.
        vLLM이 structured output을 지원해야 함.
        """
        structured_llm = self.chat.with_structured_output(schema)
        return await structured_llm.ainvoke(messages)

    def get_chat_model(self) -> ChatOpenAI:
        """ChatOpenAI 인스턴스 직접 반환 (with_structured_output 등 사용 시)."""
        return self.chat
