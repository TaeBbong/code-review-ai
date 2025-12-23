from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Any
from pydantic import Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class AdapterChatModel(BaseChatModel):
    def __init__(self, adapter: LLMAdapter, model_name: str = "adapter"):
        super().__init__()
        self._adapter = adapter
        self._model_name = model_name

    @property
    def _llm_type(self) -> str:
        return "llm_adapter"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self._model_name}

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
    @abstractmethod
    async def ainvoke(self, messages: Sequence[BaseMessage]) -> str:
        """Return raw text output from the model."""
        raise NotImplementedError
    
    def as_chat_model(self):
        return AdapterChatModel(self, model_name="adapter")
