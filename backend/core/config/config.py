from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class ReviewConfig:
    """
    - default_variant: req.variant_id가 비었을 때 사용할 prompt pack
    - packs_dir: prompt packs root directory
    - allowed_variants: 허용 목록. 비어있으면 제한 없음.
    """
    default_variant: str = "baseline"
    packs_dir: Path = Path(os.getenv("REVIEW_PROMPT_PACKS_DIR", "backend/prompts/packs"))
    allowed_variants: tuple[str, ...] = ("baseline", "g1")


def get_review_config() -> ReviewConfig:
    return ReviewConfig()
