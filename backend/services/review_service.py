from __future__ import annotations

import logging

from backend.core.settings import settings
from backend.core.prompts.registry import PromptPackRegistry
from backend.pipelines.registry import PipelineRegistry
from backend.schemas.review import ReviewRequest, ReviewResult

logger = logging.getLogger(__name__)


class ReviewService:
    """
    얇은 오케스트레이터(Facade).

    - settings만 참조
    - variant_id 결정 -> prompt pack 로드
    - preset(variant) -> pipeline 로드
    - pipeline.run(req) 실행

    내부 동작(diff 수집, chunking, tool, invoke, repair 등)은 Pipeline 구현에 있음.
    """

    def __init__(self) -> None:
        # ✅ settings만 사용
        self._prompt_registry = PromptPackRegistry(
            packs_dir=settings.review_packs_dir,
            default_variant=settings.review_default_variant,
            allowed_variants=settings.review_allowed_variants,  # tuple[str, ...] or ()
        )
        self._pipeline_registry = PipelineRegistry(
            presets_dir=str(settings.review_presets_dir),
        )

    async def review(self, req: ReviewRequest) -> ReviewResult:
        # 1) variant 결정 (pack 기준)
        variant_id = self._prompt_registry.resolve_variant(getattr(req, "variant_id", None))
        pack = self._prompt_registry.get(variant_id)

        # 2) preset -> pipeline build
        spec = self._pipeline_registry.load_spec(variant_id)
        pipeline = self._pipeline_registry.build_pipeline(spec, pack=pack)

        # 3) 실행
        logger.info(f"PIPELINE_START variant={variant_id} pack={pack.id} pipeline={spec.pipeline}")
        result = await pipeline.run(req)
        logger.info(
            "PIPELINE_DONE variant=%s repair_used=%s",
            variant_id,
            getattr(getattr(result, "meta", None), "repair_used", None),
        )
        return result
