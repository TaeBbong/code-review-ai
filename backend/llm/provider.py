from __future__ import annotations

from backend.config.settings import settings
from backend.llm.base import LLMAdapter
from backend.llm.ollama import OllamaAdapter
from backend.llm.openai_compat import OpenAICompatAdapter


def get_llm_adapter() -> LLMAdapter:
    if settings.llm_provider == "ollama":
        return OllamaAdapter(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    if settings.llm_provider == "openai_compat":
        return OpenAICompatAdapter(
            model=settings.openai_compat_model,
            base_url=settings.openai_compat_base_url,
            api_key=settings.openai_compat_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    raise ValueError(f"Unknown llm_provider: {settings.llm_provider}")
