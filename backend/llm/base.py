from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Any, Type, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class AdapterChatModel(BaseChatModel):
    def __init__(self, adapter: LLMAdapter):
        super().__init__()
        self._adapter = adapter

    @property
    def _llm_type(self) -> str:
        return "llm_adapter"

    @property
    def _identifying_params(self) -> dict:
        return {
            "provider": self._adapter.provider,
            "model": self._adapter.model_name,
        }

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = await self._adapter.ainvoke(messages)
        gen = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[gen])
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("Use async: ainvoke/_agenerate")
    

class LLMAdapter(ABC):
    @property
    @abstractmethod
    def provider(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    async def ainvoke(self, messages: Sequence[BaseMessage]) -> str:
        """Return raw text output from the model."""
        raise NotImplementedError

    async def ainvoke_structured(
        self, messages: Sequence[BaseMessage], schema: type[T]
    ) -> T:
        """
        Structured output을 사용하여 스키마에 맞는 결과 반환.
        기본 구현은 NotImplementedError. 지원하는 adapter만 오버라이드.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support structured output")

    def as_chat_model(self):
        return AdapterChatModel(self)
