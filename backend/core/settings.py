from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

    llm_provider: str = "ollama" # | "openai_compat"

    ollama_model: str = "qwen3:4b"
    ollama_base_url: str = "http://localhost:11434"

    openai_compat_base_url: str = "http://localhost:8000/v1"
    openai_compat_api_key: str = "NONEEDKEY"
    openai_compat_model: str = "local-model"

    temperature: float = 0.3
    max_tokens: int = 2048


settings = Settings()