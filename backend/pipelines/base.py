from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from backend.config.settings import settings
from backend.shared.context import run_id_var
from backend.llm.base import AdapterChatModel
from backend.llm.invoke import invoke_chain
from backend.llm.provider import get_llm_adapter
from backend.shared.parser import validate_or_repair
from backend.domain.schemas.diff import DiffChunk
from backend.domain.schemas.review import (
    ReviewRequest,
    ReviewResult,
    Issue,
    TestSuggestion,
    Question,
    PatchSuggestion,
    Summary,
    Meta,
    RiskLevel,
)


class ReviewPipeline(ABC):
    """
    Template Method: 파이프라인 큰 골격은 고정.
    variant별 변경은 hook(override)로만 한다.

    Hooks:
    - resolve_diff(): diff 준비(로컬 git 등)
    - split_chunks(): diff를 청크로 분할 (map-reduce용)
    - build_review_payload(): LLM 입력 payload 확장
    - reduce_results(): 여러 리뷰 결과를 병합 (map-reduce용)
    - after_run(): 결과 저장/리포트 등 후처리
    """

    def __init__(self, *, pack, params: Dict[str, Any]):
        self.pack = pack
        self.params = params

    async def run(self, req: ReviewRequest) -> ReviewResult:
        run_id = run_id_var.get()

        # 1) diff 준비 (hook)
        diff, diff_target = await self.resolve_diff(req)

        # 2) 청크 분할 (hook) - 기본은 단일 청크
        chunks = await self.split_chunks(diff)

        # 3) llm + parser 준비
        adapter = get_llm_adapter()
        llm = AdapterChatModel(adapter)
        parser = PydanticOutputParser(pydantic_object=ReviewResult)
        format_instructions = parser.get_format_instructions()

        # 4) chains 구성
        review_chain = self.build_chain(llm, mode="review", format_instructions=format_instructions)
        repair_chain = self.build_chain(llm, mode="repair", format_instructions=format_instructions)

        # 5) 리뷰 실행 (단일 vs map-reduce)
        bad_max_chars = int(self.params.get("bad_max_chars", self.pack.params.get("bad_max_chars", 4000)))

        if len(chunks) <= 1:
            # 단일 청크: 기존 로직
            chunk_diff = chunks[0].content if chunks else diff
            result, repair_used, raw_text, raw_json, fixed_json = await self._review_single(
                req=req,
                diff=chunk_diff,
                diff_target=diff_target,
                adapter=adapter,
                review_chain=review_chain,
                repair_chain=repair_chain,
                bad_max_chars=bad_max_chars,
            )
        else:
            # 다중 청크: map-reduce
            result, repair_used, raw_text, raw_json, fixed_json = await self._review_map_reduce(
                req=req,
                chunks=chunks,
                diff_target=diff_target,
                adapter=adapter,
                review_chain=review_chain,
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
            raw_text=raw_text,
            raw_json=raw_json,
            fixed_json=fixed_json,
        )
        return result

    async def _review_single(
        self,
        *,
        req: ReviewRequest,
        diff: str,
        diff_target: str,
        adapter,
        review_chain: Runnable,
        repair_chain: Runnable,
        bad_max_chars: int,
    ) -> tuple[ReviewResult, bool, str, str | None, str | None]:
        """단일 diff에 대한 리뷰 실행."""
        payload = await self.build_review_payload(req=req, diff=diff, diff_target=diff_target)

        # Structured output 사용 시
        if settings.use_structured_output:
            messages = review_chain.first.format_messages(**payload)
            result = await adapter.ainvoke_structured(messages, ReviewResult)
            return result, False, "", None, None

        # 기존 방식: 텍스트 파싱 + repair
        msg = await invoke_chain(review_chain, payload)
        content = msg.content or ""

        result, repair_used, raw_json, fixed_json = await validate_or_repair(
            raw_text=content,
            repair_chain=repair_chain,
            bad_max_chars=bad_max_chars,
        )
        return result, repair_used, content, raw_json, fixed_json

    async def _review_map_reduce(
        self,
        *,
        req: ReviewRequest,
        chunks: List[DiffChunk],
        diff_target: str,
        adapter,
        review_chain: Runnable,
        repair_chain: Runnable,
        bad_max_chars: int,
    ) -> tuple[ReviewResult, bool, str, str | None, str | None]:
        """Map-Reduce: 청크별 리뷰 후 병합."""
        max_concurrency = int(self.params.get("max_concurrency", 4))
        semaphore = asyncio.Semaphore(max_concurrency)

        async def review_chunk(chunk: DiffChunk) -> ReviewResult:
            async with semaphore:
                payload = await self.build_review_payload(
                    req=req,
                    diff=chunk.content,
                    diff_target=diff_target,
                    chunk=chunk,
                )

                # Structured output 사용 시
                if settings.use_structured_output:
                    messages = review_chain.first.format_messages(**payload)
                    return await adapter.ainvoke_structured(messages, ReviewResult)

                # 기존 방식
                msg = await invoke_chain(review_chain, payload)
                content = msg.content or ""

                result, _, _, _ = await validate_or_repair(
                    raw_text=content,
                    repair_chain=repair_chain,
                    bad_max_chars=bad_max_chars,
                )
                return result

        tasks = [review_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 실패한 청크는 제외
        valid_results = [r for r in results if isinstance(r, ReviewResult)]

        # Reduce: 결과 병합 (hook)
        merged = await self.reduce_results(valid_results)

        # repair_used는 하나라도 True면 True
        repair_used = any(getattr(r.meta, "repair_used", False) for r in valid_results)

        return merged, repair_used, "", None, None

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

    async def split_chunks(self, diff: str) -> List[DiffChunk]:
        """
        diff를 청크로 분할. 기본 구현은 분할 없이 단일 청크 반환.

        오버라이드하여 파일별, 함수별 등으로 분할 가능.
        단일 청크 반환 시 기존 로직대로 동작.
        """
        return [DiffChunk(file_path="", content=diff)]

    async def build_review_payload(
        self,
        *,
        req: ReviewRequest,
        diff: str,
        diff_target: str,
        chunk: DiffChunk | None = None,
    ) -> Dict[str, Any]:
        """
        LLM에 전달할 payload 구성.
        chunk가 제공되면 map 단계에서 호출됨.
        """
        payload = {
            "variant_id": getattr(req, "variant_id", None),
            "diff": diff,
            "diff_target": diff_target,
        }
        if chunk and chunk.file_path:
            payload["file_path"] = chunk.file_path
        return payload

    async def reduce_results(self, results: List[ReviewResult]) -> ReviewResult:
        """
        여러 리뷰 결과를 하나로 병합.
        기본 구현: 모든 이슈/제안을 합치고 summary는 재구성.
        """
        if not results:
            return ReviewResult()

        if len(results) == 1:
            return results[0]

        # 모든 항목 수집
        all_issues: List[Issue] = []
        all_test_suggestions: List[TestSuggestion] = []
        all_questions: List[Question] = []
        all_blockers: List[str] = []
        all_patches: List[PatchSuggestion] = []
        all_key_points: List[str] = []

        for r in results:
            all_issues.extend(r.issues)
            all_test_suggestions.extend(r.test_suggestions)
            all_questions.extend(r.questions_to_author)
            all_blockers.extend(r.merge_blockers)
            all_patches.extend(r.patch_suggestions)
            all_key_points.extend(r.summary.key_points)

        # 이슈 ID 재할당 (중복 방지)
        for i, issue in enumerate(all_issues, 1):
            issue.id = f"ISS-{i:03d}"

        # overall_risk: 가장 높은 위험도 사용
        risk_priority = {RiskLevel.high: 3, RiskLevel.medium: 2, RiskLevel.low: 1}
        max_risk = RiskLevel.low
        for r in results:
            if risk_priority.get(r.summary.overall_risk, 0) > risk_priority.get(max_risk, 0):
                max_risk = r.summary.overall_risk

        merged = ReviewResult(
            meta=Meta(),
            summary=Summary(
                intent=f"Review of {len(results)} file(s)",
                overall_risk=max_risk,
                key_points=all_key_points[:10],
            ),
            issues=all_issues,
            test_suggestions=all_test_suggestions,
            questions_to_author=all_questions,
            merge_blockers=list(set(all_blockers)),
            patch_suggestions=all_patches,
        )

        return merged

    def build_chain(self, llm: AdapterChatModel, *, mode: str, format_instructions: str) -> Runnable:
        """PromptPack + format_instructions로 chain 구성."""
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
        """후처리 hook."""
        return
