from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from backend.shared.context import run_id_var
from backend.llm.base import AdapterChatModel
from backend.llm.invoke import invoke_chain
from backend.llm.provider import get_llm_adapter
from backend.shared.parser import validate_or_repair
from backend.domain.schemas.review import ReviewRequest, ReviewResult


class ReviewPipeline(ABC):
    """
    Template Method: 파이프라인 큰 골격은 고정.
    variant별 변경은 hook(override)로만 한다.

    - resolve_diff(): diff 준비(로컬 git 등)
    - build_review_payload(): LLM 입력 payload 확장
    - after_run(): 결과 저장/리포트 등 후처리
    """

    def __init__(self, *, pack, params: Dict[str, Any]):
        self.pack = pack
        self.params = params

    async def run(self, req: ReviewRequest) -> ReviewResult:
        run_id = run_id_var.get()

        # 1) diff 준비 (hook)
        diff, diff_target = await self.resolve_diff(req)

        # 2) llm + parser
        adapter = get_llm_adapter()
        llm = AdapterChatModel(adapter)

        parser = PydanticOutputParser(pydantic_object=ReviewResult)
        format_instructions = parser.get_format_instructions()

        # 3) chains (공통)
        review_chain = self.build_chain(llm, mode="review", format_instructions=format_instructions)
        repair_chain = self.build_chain(llm, mode="repair", format_instructions=format_instructions)

        # 4) invoke
        payload = await self.build_review_payload(req=req, diff=diff, diff_target=diff_target)
        msg = await invoke_chain(review_chain, payload)
        content = msg.content or ""

        # 5) validate or repair
        bad_max_chars = int(self.params.get("bad_max_chars", self.pack.params.get("bad_max_chars", 4000)))
        result, repair_used, raw_json, fixed_json = await validate_or_repair(
            raw_text=content,
            repair_chain=repair_chain,
            bad_max_chars=bad_max_chars,
        )

        # 6) meta inject
        result.meta.variant_id = getattr(req, "variant_id", None) or ""
        result.meta.run_id = run_id
        result.meta.repair_used = repair_used
        result.meta.llm_provider = adapter.provider
        result.meta.model = adapter.model_name
        result.meta.diff_target = diff_target
        result.meta.generated_at = datetime.now().isoformat()

        # 7) 후처리 hook
        await self.after_run(
            req=req,
            result=result,
            raw_text=content,
            raw_json=raw_json,
            fixed_json=fixed_json,
        )
        return result

    # --------------------
    # Hooks
    # --------------------
    @abstractmethod
    async def resolve_diff(self, req: ReviewRequest) -> tuple[str, str]:
        """
        Returns: (diff_text, diff_target_label)
        diff_target_label 예: "raw" | "staged" | "worktree" | "A..B"
        """
        raise NotImplementedError

    async def build_review_payload(self, *, req: ReviewRequest, diff: str, diff_target: str) -> Dict[str, Any]:
        """
        기본 payload. variant별로 확장 가능(evidence_pack 등).
        """
        return {
            "variant_id": getattr(req, "variant_id", None),
            "diff": diff,
            "diff_target": diff_target,
        }

    def build_chain(self, llm: AdapterChatModel, *, mode: str, format_instructions: str) -> Runnable:
        """
        PromptPack + format_instructions로 chain 구성.
        review_service.py에 의존하지 않도록 여기서 직접 만든다.
        """
        if mode == "review":
            system, user = self.pack.review_system, self.pack.review_user
        elif mode == "repair":
            system, user = self.pack.repair_system, self.pack.repair_user
        else:
            raise ValueError(f"Unknown mode: {mode}")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", user),
            ]
        )
        # format_instructions는 prompt 템플릿에서 {format_instructions}로 사용 가능
        return prompt.partial(format_instructions=format_instructions) | llm

    async def after_run(
        self,
        *,
        req: ReviewRequest,
        result: ReviewResult,
        raw_text: str,
        raw_json: str | None,
        fixed_json: str | None,
    ) -> None:
        return
