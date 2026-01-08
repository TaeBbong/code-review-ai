from pathlib import Path

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
    use_structured_output: bool = False  # vLLM structured output 사용 여부

    review_default_variant: str = "G0-baseline"
    review_packs_dir: Path = Path("backend/domain/prompts/packs")
    review_presets_dir: Path = Path("backend/pipelines/presets")
    review_allowed_variants_raw: str = ""
    review_repo_path: Path | None = None

    @property
    def review_allowed_variants(self) -> tuple[str, ...]:
        raw = (self.review_allowed_variants_raw or "").strip()
        if not raw:
            return None
        return tuple([x.strip() for x in raw.split(",") if x.strip()])


settings = Settings()