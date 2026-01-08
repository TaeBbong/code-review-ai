from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.config.settings import settings
from backend.pipelines.base import ReviewPipeline
from backend.llm.base import AdapterChatModel
from backend.llm.invoke import invoke_chain
from backend.llm.provider import get_llm_adapter
from backend.shared.parser import validate_or_repair
from backend.domain.tools.git_diff import get_git_diff, GitError
from backend.domain.schemas.review import (
    ReviewResult,
    Issue,
    Summary,
    Meta,
    RiskLevel,
)


class IterativeRefinementPipeline(ReviewPipeline):
    """
    G2-iterative: 1차 리뷰 → 자기 검증 → 오탐 필터링 → 최종 리뷰.

    핵심 기능:
    - 1차 리뷰: 일반적인 코드 리뷰 수행
    - 자기 검증: 1차 리뷰 결과를 다시 LLM에게 보내 오탐(false positive) 검증
    - 오탐 필터링: 검증 결과를 바탕으로 신뢰도 낮은 이슈 제거
    - False positive 감소에 효과적
    """

    async def resolve_diff(self, req):
        raw = (getattr(req, "diff", None) or "").strip()
        if raw:
            return raw, "raw"

        diff_target = (getattr(req, "diff_target", None) or self.params.get("diff_source") or "staged").strip()
        repo_path = str(settings.review_repo_path) if settings.review_repo_path else None

        try:
            diff = get_git_diff(
                diff_target=diff_target,
                repo_path=repo_path,
                context_lines=int(self.params.get("context_lines", 3)),
                max_chars=int(self.params.get("max_chars", 1_500_000)),
            )
        except GitError:
            diff = ""
        return diff, diff_target

    async def run(self, req) -> ReviewResult:
        """
        Template Method의 run()을 오버라이드하여 iterative refinement 로직 구현.
        """
        from backend.shared.context import run_id_var
        from datetime import datetime

        run_id = run_id_var.get()

        # 1) diff 준비
        diff, diff_target = await self.resolve_diff(req)

        if not diff.strip():
            return ReviewResult(
                meta=Meta(
                    variant_id=getattr(req, "variant_id", None) or "",
                    run_id=run_id,
                    diff_target=diff_target,
                    generated_at=datetime.now().isoformat(),
                ),
                summary=Summary(
                    intent="No changes to review",
                    overall_risk=RiskLevel.low,
                    key_points=[],
                ),
            )

        # 2) LLM + parser 준비
        adapter = get_llm_adapter()
        llm = AdapterChatModel(adapter)
        parser = PydanticOutputParser(pydantic_object=ReviewResult)
        format_instructions = parser.get_format_instructions()
        bad_max_chars = int(self.params.get("bad_max_chars", self.pack.params.get("bad_max_chars", 4000)))

        # 3) 1차 리뷰
        initial_result = await self._initial_review(
            req=req,
            diff=diff,
            diff_target=diff_target,
            llm=llm,
            format_instructions=format_instructions,
            bad_max_chars=bad_max_chars,
        )

        # 4) 자기 검증 (이슈가 있는 경우만)
        if initial_result.issues:
            refined_result = await self._refine_review(
                req=req,
                diff=diff,
                initial_result=initial_result,
                llm=llm,
                format_instructions=format_instructions,
                bad_max_chars=bad_max_chars,
            )
        else:
            refined_result = initial_result

        # 5) meta inject
        refined_result.meta.variant_id = getattr(req, "variant_id", None) or ""
        refined_result.meta.run_id = run_id
        refined_result.meta.llm_provider = adapter.provider
        refined_result.meta.model = adapter.model_name
        refined_result.meta.diff_target = diff_target
        refined_result.meta.generated_at = datetime.now().isoformat()

        # 6) 후처리
        await self.after_run(
            req=req,
            result=refined_result,
            raw_text="",
            raw_json=None,
            fixed_json=None,
        )

        return refined_result

    async def _initial_review(
        self,
        *,
        req,
        diff: str,
        diff_target: str,
        llm: AdapterChatModel,
        format_instructions: str,
        bad_max_chars: int,
    ) -> ReviewResult:
        """1차 리뷰 수행."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.pack.review_system),
            ("human", self.pack.review_user),
        ])
        chain = prompt.partial(format_instructions=format_instructions) | llm

        repair_prompt = ChatPromptTemplate.from_messages([
            ("system", self.pack.repair_system),
            ("human", self.pack.repair_user),
        ])
        repair_chain = repair_prompt.partial(format_instructions=format_instructions) | llm

        payload = await self.build_review_payload(
            req=req,
            diff=diff,
            diff_target=diff_target,
        )

        msg = await invoke_chain(chain, payload)
        content = msg.content or ""

        result, _, _, _ = await validate_or_repair(
            raw_text=content,
            repair_chain=repair_chain,
            bad_max_chars=bad_max_chars,
        )

        return result

    async def _refine_review(
        self,
        *,
        req,
        diff: str,
        initial_result: ReviewResult,
        llm: AdapterChatModel,
        format_instructions: str,
        bad_max_chars: int,
    ) -> ReviewResult:
        """
        자기 검증: 1차 리뷰 결과를 다시 검토하여 오탐 필터링.
        """
        # 1차 리뷰 이슈를 텍스트로 변환
        issues_text = self._format_issues_for_refinement(initial_result.issues)

        # refinement 프롬프트 구성
        refinement_system = """You are a code review validator.
Your job is to review the initial findings and remove ONLY obvious false positives.

Keep an issue UNLESS it is clearly wrong:
- The issue does not exist in the diff at all
- The issue is about code that wasn't changed
- The issue is completely irrelevant to the changes

When in doubt, KEEP the issue. It's better to have a few false positives than miss real bugs.

Return ONLY JSON. No markdown, no commentary."""

        refinement_user = """Review the initial findings and filter out only the obvious false positives.

{format_instructions}

## Original Diff
{diff}

## Initial Review Findings
{issues_text}

Rules:
- KEEP most issues - only remove ones that are clearly wrong
- When uncertain, keep the issue
- You may adjust severity levels if needed
- Keep the same JSON structure as the original review
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", refinement_system),
            ("human", refinement_user),
        ])
        chain = prompt.partial(format_instructions=format_instructions) | llm

        repair_prompt = ChatPromptTemplate.from_messages([
            ("system", self.pack.repair_system),
            ("human", self.pack.repair_user),
        ])
        repair_chain = repair_prompt.partial(format_instructions=format_instructions) | llm

        payload = {
            "diff": diff,
            "issues_text": issues_text,
            "variant_id": getattr(req, "variant_id", None),
        }

        msg = await invoke_chain(chain, payload)
        content = msg.content or ""

        refined_result, _, _, _ = await validate_or_repair(
            raw_text=content,
            repair_chain=repair_chain,
            bad_max_chars=bad_max_chars,
        )

        # 필터링된 이슈 수 추적
        initial_count = len(initial_result.issues)
        refined_count = len(refined_result.issues)
        filtered_count = initial_count - refined_count

        # summary에 필터링 정보 추가
        if filtered_count > 0:
            refined_result.summary.key_points.insert(
                0, f"Filtered {filtered_count} potential false positive(s) from initial {initial_count} issue(s)"
            )

        return refined_result

    def _format_issues_for_refinement(self, issues: List[Issue]) -> str:
        """이슈 목록을 검증용 텍스트로 변환."""
        if not issues:
            return "No issues found."

        lines = []
        for issue in issues:
            locations = ", ".join(
                f"{loc.file}:{loc.line_start}-{loc.line_end}"
                for loc in (issue.locations or [])
            ) or "unknown location"

            lines.append(f"""
Issue ID: {issue.id}
Title: {issue.title}
Severity: {issue.severity.value if issue.severity else 'unknown'}
Category: {issue.category or 'unknown'}
Location: {locations}
Description: {issue.description}
Suggested Fix: {issue.suggested_fix or 'N/A'}
""")

        return "\n---\n".join(lines)
