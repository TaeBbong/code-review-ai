"""
G4-MultiReview Pipeline

논문 기반 전략:
- 동일 프롬프트로 N회 독립 리뷰 실행
- LLM을 사용하여 결과 집계 (Aggregation)
- F1 점수 43.67% 향상, Recall 118.83% 향상 입증

핵심 원리:
- LLM의 리뷰 결과는 실행마다 다른 이슈를 탐지 (무작위성)
- 여러 번 실행 후 집계하면 더 많은 실제 결함 탐지 (Recall ↑)
- 집계 과정에서 반복 확인된 이슈는 신뢰도 높음 (Precision ↑)

Aggregation Modes:
- llm: LLM 기반 semantic 중복 제거 (비용 높음, 정확도 높음)
- voting: 투표 기반 필터링 (비용 없음, FP 감소에 효과적)
- simple: 단순 합산 (fallback)
"""

from __future__ import annotations

import asyncio
from typing import List

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
    TestSuggestion,
    Question,
    PatchSuggestion,
    Summary,
    Meta,
    RiskLevel,
)


class MultiReviewPipeline(ReviewPipeline):
    """
    G4-multireview: 동일 diff에 대해 N회 독립 리뷰 후 LLM 집계.

    핵심 기능:
    - 동일 프롬프트로 여러 번 독립적으로 리뷰 수행
    - 각 리뷰는 서로 다른 이슈를 탐지할 수 있음 (LLM 무작위성 활용)
    - 집계 단계에서 중복 제거 및 신뢰도 기반 필터링
    - 논문에서 입증된 Multi-Review Aggregation 전략 구현

    Attributes:
        last_round_results: 마지막 실행의 각 라운드별 리뷰 결과 (평가/디버깅용)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_round_results: List[ReviewResult] = []

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
        Template Method의 run()을 오버라이드하여 multi-review 로직 구현.
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

        # 3) N회 독립 리뷰 수행
        num_reviews = int(self.params.get("num_reviews", 3))
        max_concurrency = int(self.params.get("max_concurrency", 3))

        review_results = await self._run_multiple_reviews(
            req=req,
            diff=diff,
            diff_target=diff_target,
            llm=llm,
            format_instructions=format_instructions,
            bad_max_chars=bad_max_chars,
            num_reviews=num_reviews,
            max_concurrency=max_concurrency,
        )

        # 라운드별 결과 저장 (평가/디버깅용)
        self.last_round_results = review_results

        # 4) 결과 집계 (Aggregation)
        aggregation_mode = self.params.get("aggregation_mode", "llm")

        if aggregation_mode == "voting" and len(review_results) > 1:
            # 투표 기반 집계: min_votes 이상 발견된 이슈만 포함
            min_votes = int(self.params.get("min_votes", 2))
            aggregated = self._aggregate_voting(
                review_results=review_results,
                min_votes=min_votes,
            )
        elif aggregation_mode == "llm" and len(review_results) > 1:
            # LLM 기반 집계: 중복 제거 + 신뢰도 평가
            aggregated = await self._aggregate_with_llm(
                review_results=review_results,
                diff=diff,
                llm=llm,
                format_instructions=format_instructions,
                bad_max_chars=bad_max_chars,
            )
        else:
            # 단순 합산 집계 (fallback)
            aggregated = self._aggregate_simple(review_results)

        # 5) meta inject
        aggregated.meta.variant_id = getattr(req, "variant_id", None) or ""
        aggregated.meta.run_id = run_id
        aggregated.meta.llm_provider = adapter.provider
        aggregated.meta.model = adapter.model_name
        aggregated.meta.diff_target = diff_target
        aggregated.meta.generated_at = datetime.now().isoformat()

        # 6) 통계 정보 추가
        total_issues_before_agg = sum(len(r.issues) for r in review_results)
        aggregated.summary.key_points.insert(
            0, f"Aggregated from {len(review_results)} independent reviews ({total_issues_before_agg} raw issues → {len(aggregated.issues)} final)"
        )

        # 7) 후처리
        await self.after_run(
            req=req,
            result=aggregated,
            raw_text="",
            raw_json=None,
            fixed_json=None,
        )

        return aggregated

    async def _run_multiple_reviews(
        self,
        *,
        req,
        diff: str,
        diff_target: str,
        llm: AdapterChatModel,
        format_instructions: str,
        bad_max_chars: int,
        num_reviews: int,
        max_concurrency: int,
    ) -> List[ReviewResult]:
        """N회 독립 리뷰 수행."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def single_review(review_idx: int) -> ReviewResult:
            async with semaphore:
                # 리뷰 프롬프트 구성
                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.pack.review_system),
                    ("human", self.pack.review_user),
                ])
                chain = prompt.partial(format_instructions=format_instructions) | llm

                # repair chain
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
                payload["review_pass"] = review_idx + 1

                msg = await invoke_chain(chain, payload)
                content = msg.content or ""

                result, _, _, _ = await validate_or_repair(
                    raw_text=content,
                    repair_chain=repair_chain,
                    bad_max_chars=bad_max_chars,
                )

                # 이슈에 리뷰 패스 정보 추가 (메타데이터용)
                for issue in result.issues:
                    if not issue.evidence_ids:
                        issue.evidence_ids = []
                    issue.evidence_ids.append(f"review_pass_{review_idx + 1}")

                return result

        tasks = [single_review(i) for i in range(num_reviews)]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 성공한 결과만 필터링
        valid_results = [
            r for r in raw_results
            if isinstance(r, ReviewResult)
        ]

        return valid_results

    async def _aggregate_with_llm(
        self,
        *,
        review_results: List[ReviewResult],
        diff: str,
        llm: AdapterChatModel,
        format_instructions: str,
        bad_max_chars: int,
    ) -> ReviewResult:
        """
        LLM 기반 집계: 여러 리뷰 결과를 분석하여 중복 제거 및 신뢰도 평가.

        논문 전략:
        - 여러 리뷰에서 반복 발견된 이슈는 신뢰도 높음
        - 한 번만 발견된 이슈도 유효할 수 있으므로 완전히 제거하지 않음
        - LLM이 의미적 중복을 판단하여 병합
        """
        # 모든 이슈를 텍스트로 변환
        all_issues_text = self._format_all_issues_for_aggregation(review_results)

        aggregation_system = """You are a code review aggregator.
You have received multiple independent reviews of the same code changes.
Your job is to merge these reviews into a single, high-quality review report.

Guidelines:
1. MERGE duplicate issues: If multiple reviews found the same or very similar issue, combine them into one.
2. INCREASE confidence for repeated issues: Issues found by multiple reviews are more likely to be real.
3. KEEP unique issues: Even if an issue was found by only one review, include it if it seems valid.
4. REMOVE obvious false positives: Only remove issues that are clearly wrong when cross-checking with others.
5. PRIORITIZE: Focus on functional bugs (logic errors, null checks, API issues) over style issues.

Return ONLY JSON. No markdown, no commentary."""

        aggregation_user = """Aggregate the following independent review results into a single comprehensive review.

{format_instructions}

## Original Diff
{diff}

## Independent Review Results
{all_issues_text}

Instructions:
- Merge semantically duplicate issues (same bug, different wording)
- For merged issues, note how many reviews found it (e.g., "Found by 3/5 reviews")
- Keep unique valid issues even if only one review found them
- Assign higher confidence (0.7+) to issues found by multiple reviews
- Assign lower confidence (0.3-0.5) to issues found by only one review
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", aggregation_system),
            ("human", aggregation_user),
        ])
        chain = prompt.partial(format_instructions=format_instructions) | llm

        repair_prompt = ChatPromptTemplate.from_messages([
            ("system", self.pack.repair_system),
            ("human", self.pack.repair_user),
        ])
        repair_chain = repair_prompt.partial(format_instructions=format_instructions) | llm

        payload = {
            "diff": diff,
            "all_issues_text": all_issues_text,
        }

        msg = await invoke_chain(chain, payload)
        content = msg.content or ""

        aggregated, _, _, _ = await validate_or_repair(
            raw_text=content,
            repair_chain=repair_chain,
            bad_max_chars=bad_max_chars,
        )

        # 이슈 ID 재할당
        for i, issue in enumerate(aggregated.issues, 1):
            issue.id = f"ISS-{i:03d}"

        return aggregated

    def _aggregate_voting(
        self,
        review_results: List[ReviewResult],
        min_votes: int = 2,
    ) -> ReviewResult:
        """
        투표 기반 집계: min_votes 이상 발견된 이슈만 포함.

        이슈 유사도 판단 기준:
        - 같은 파일 + 같은 라인 범위 (±3) + 같은 카테고리 → 동일 이슈로 간주
        - 파일/라인 정보 없으면 카테고리 + 제목 키워드 유사도로 판단

        Args:
            review_results: N회 독립 리뷰 결과
            min_votes: 최소 투표 수 (이 이상 발견된 이슈만 포함)
        """
        if not review_results:
            return ReviewResult()

        # 1) 모든 이슈 수집 + 투표 카운트
        issue_votes: dict[str, list[Issue]] = {}  # signature -> [issues]

        for result in review_results:
            for issue in result.issues:
                sig = self._get_issue_signature(issue)
                if sig not in issue_votes:
                    issue_votes[sig] = []
                issue_votes[sig].append(issue)

        # 2) min_votes 이상인 이슈만 필터링
        filtered_issues: List[Issue] = []
        for sig, issues in issue_votes.items():
            vote_count = len(issues)
            if vote_count >= min_votes:
                # 대표 이슈 선택 (첫 번째) + 투표 정보 추가
                representative = issues[0].model_copy(deep=True)
                representative.title = f"[{vote_count}/{len(review_results)} votes] {representative.title or ''}"
                filtered_issues.append(representative)

        # 3) severity 기준 정렬 (high → medium → low)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        filtered_issues.sort(
            key=lambda x: severity_order.get(x.severity.value if x.severity else "low", 2)
        )

        # 4) 이슈 ID 재할당
        for i, issue in enumerate(filtered_issues, 1):
            issue.id = f"ISS-{i:03d}"

        # 5) overall_risk 계산
        risk_priority = {RiskLevel.high: 3, RiskLevel.medium: 2, RiskLevel.low: 1}
        max_risk = RiskLevel.low
        for result in review_results:
            if risk_priority.get(result.summary.overall_risk, 0) > risk_priority.get(max_risk, 0):
                max_risk = result.summary.overall_risk

        # 6) 기타 정보 수집
        all_test_suggestions: List[TestSuggestion] = []
        all_questions: List[Question] = []
        all_blockers: List[str] = []
        all_patches: List[PatchSuggestion] = []

        for result in review_results:
            all_test_suggestions.extend(result.test_suggestions)
            all_questions.extend(result.questions_to_author)
            all_blockers.extend(result.merge_blockers)
            all_patches.extend(result.patch_suggestions)

        total_before = sum(len(r.issues) for r in review_results)

        return ReviewResult(
            meta=Meta(),
            summary=Summary(
                intent=f"Voting aggregation (min_votes={min_votes})",
                overall_risk=max_risk,
                key_points=[
                    f"Filtered {total_before} raw issues → {len(filtered_issues)} "
                    f"(kept issues with {min_votes}+ votes)"
                ],
            ),
            issues=filtered_issues,
            test_suggestions=all_test_suggestions[:5],
            questions_to_author=all_questions[:3],
            merge_blockers=list(set(all_blockers)),
            patch_suggestions=all_patches[:5],
        )

    def _get_issue_signature(self, issue: Issue) -> str:
        """
        이슈의 고유 시그니처 생성 (유사 이슈 그룹핑용).

        시그니처 = 카테고리 + 파일 + 라인범위(10단위 버킷)
        """
        category = issue.category.value if issue.category else "unknown"

        # 위치 정보 추출
        file_path = ""
        line_bucket = 0
        if issue.locations:
            loc = issue.locations[0]
            file_path = loc.file or ""
            # 라인 번호를 10단위 버킷으로 그룹핑 (예: 15 → 10, 27 → 20)
            line_bucket = (loc.line_start or 0) // 10 * 10

        # 파일/라인 정보 없으면 제목 키워드 사용
        if not file_path:
            # 제목에서 주요 키워드 추출 (소문자, 정렬)
            title_words = sorted(
                word.lower() for word in (issue.title or "").split()
                if len(word) > 3
            )[:3]
            return f"{category}::{':'.join(title_words)}"

        return f"{category}::{file_path}::{line_bucket}"

    def _aggregate_simple(self, review_results: List[ReviewResult]) -> ReviewResult:
        """
        단순 합산 집계 (fallback): 모든 이슈를 수집하고 기본 중복 제거.
        """
        if not review_results:
            return ReviewResult()

        all_issues: List[Issue] = []
        all_test_suggestions: List[TestSuggestion] = []
        all_questions: List[Question] = []
        all_blockers: List[str] = []
        all_patches: List[PatchSuggestion] = []
        all_key_points: List[str] = []

        for idx, result in enumerate(review_results, 1):
            for issue in result.issues:
                # 이슈 제목에 리뷰 패스 정보 추가
                issue.title = f"[Pass {idx}] {issue.title or ''}".strip()
                all_issues.append(issue)

            all_test_suggestions.extend(result.test_suggestions)
            all_questions.extend(result.questions_to_author)
            all_blockers.extend(result.merge_blockers)
            all_patches.extend(result.patch_suggestions)
            all_key_points.extend(result.summary.key_points)

        # 이슈 ID 재할당
        for i, issue in enumerate(all_issues, 1):
            issue.id = f"ISS-{i:03d}"

        # overall_risk: 가장 높은 위험도 사용
        risk_priority = {RiskLevel.high: 3, RiskLevel.medium: 2, RiskLevel.low: 1}
        max_risk = RiskLevel.low
        for result in review_results:
            if risk_priority.get(result.summary.overall_risk, 0) > risk_priority.get(max_risk, 0):
                max_risk = result.summary.overall_risk

        merged = ReviewResult(
            meta=Meta(),
            summary=Summary(
                intent=f"Multi-review aggregation from {len(review_results)} independent reviews",
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

    def _format_all_issues_for_aggregation(self, review_results: List[ReviewResult]) -> str:
        """모든 리뷰 결과의 이슈를 집계용 텍스트로 변환."""
        sections = []

        for idx, result in enumerate(review_results, 1):
            section_lines = [f"### Review Pass {idx}"]
            section_lines.append(f"Risk Level: {result.summary.overall_risk.value if result.summary.overall_risk else 'unknown'}")
            section_lines.append(f"Issues Found: {len(result.issues)}")
            section_lines.append("")

            for issue in result.issues:
                locations = ", ".join(
                    f"{loc.file}:{loc.line_start}-{loc.line_end}"
                    for loc in (issue.locations or [])
                ) or "unknown location"

                section_lines.append(f"""
Issue: {issue.title}
Severity: {issue.severity.value if issue.severity else 'unknown'}
Category: {issue.category.value if issue.category else 'unknown'}
Location: {locations}
Description: {issue.description}
Suggested Fix: {issue.suggested_fix or 'N/A'}
""")

            sections.append("\n".join(section_lines))

        return "\n\n---\n\n".join(sections)
