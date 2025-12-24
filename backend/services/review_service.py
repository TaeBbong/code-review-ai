from __future__ import annotations

import logging
from datetime import datetime

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from backend.core.context import run_id_var
from backend.core.config.config import get_review_config
from backend.core.llm.base import AdapterChatModel
from backend.core.llm.invoke import invoke_chain
from backend.core.llm.provider import get_llm_adapter
from backend.core.parsing.validate_repair import validate_or_repair
from backend.core.prompts.registry import PromptPack, PromptPackRegistry
from backend.schemas.review import ReviewRequest, ReviewResult

logger = logging.getLogger(__name__)


def build_chain_from_pack(
        llm: AdapterChatModel, 
        pack: PromptPack, 
        format_instructions: str, 
        mode: str
    ) -> Runnable:
    """
    mode: "review" | "repair"
    """
    if mode == "review":
        system, user = pack.review_system, pack.review_user
    elif mode == "repair":
        system, user = pack.repair_system, pack.repair_user
    else:
        raise ValueError(f"Unknown mode: {mode}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", user),
    ])
    return prompt.partial(format_instructions=format_instructions) | llm


class ReviewService:
    async def review(self, req: ReviewRequest) -> ReviewResult:
        run_id = run_id_var.get()

        # 1) config + prompt pack
        cfg = get_review_config()
        registry = PromptPackRegistry(
            packs_dir=cfg.packs_dir,
            default_variant=cfg.default_variant,
            allowed_variants=cfg.allowed_variants,
        )
        variant_id = registry.resolve_variant(getattr(req, "variant_id", None))
        pack = registry.get(variant_id)

        # 2) llm + parser
        adapter = get_llm_adapter()
        llm = AdapterChatModel(adapter)

        parser = PydanticOutputParser(pydantic_object=ReviewResult)
        format_instructions = parser.get_format_instructions()

        # 3) chains
        review_chain = build_chain_from_pack(llm, pack, format_instructions, mode="review")
        repair_chain = build_chain_from_pack(llm, pack, format_instructions, mode="repair")

        # 4) invoke
        msg = await invoke_chain(review_chain, {"variant_id": variant_id, "diff": req.diff})
        content = msg.content or ""
        logger.info(
            f"LLM_OUTPUT run_id={run_id} variant={variant_id} pack={pack.id} len={len(content)}",
        )

        # 5) validate or repair
        bad_max_chars = int(pack.params.get("bad_max_chars", 4000))
        result, repair_used, raw_json, fixed_json = await validate_or_repair(
            raw_text=content,
            repair_chain=repair_chain,
            bad_max_chars=bad_max_chars,
        )

        if repair_used:
            logger.warning(
                f"LLM_REPAIR_USED run_id={run_id} variant={variant_id}",
            )

        # 6) meta inject
        result.meta.variant_id = variant_id
        result.meta.run_id = run_id
        result.meta.repair_used = repair_used
        result.meta.llm_provider = adapter.provider
        result.meta.model = adapter.model_name
        result.meta.diff_target = "raw"
        result.meta.generated_at = datetime.now().isoformat()

        logger.info(
            f"DONE run_id={run_id} variant={variant_id} repair_used={repair_used}",
        )
        return result
