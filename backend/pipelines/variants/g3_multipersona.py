from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
    TestSuggestion,
    Question,
    PatchSuggestion,
    Summary,
    Meta,
    RiskLevel,
)


@dataclass
class Persona:
    """리뷰어 페르소나 정의."""
    id: str
    name: str
    focus: str
    system_prompt: str


# 기본 페르소나 정의
DEFAULT_PERSONAS = [
    Persona(
        id="security",
        name="Security Reviewer",
        focus="security vulnerabilities",
        system_prompt="""You are a security-focused code reviewer.
Your job is to find security vulnerabilities, potential exploits, and unsafe patterns.

Focus areas:
- Injection attacks (SQL, command, XSS, etc.)
- Authentication and authorization issues
- Sensitive data exposure
- Input validation problems
- Cryptographic weaknesses
- OWASP Top 10 vulnerabilities

Return ONLY JSON. No markdown, no commentary.
Only report issues you are confident about from the diff.""",
    ),
    Persona(
        id="performance",
        name="Performance Reviewer",
        focus="performance issues",
        system_prompt="""You are a performance-focused code reviewer.
Your job is to find performance bottlenecks, inefficiencies, and scalability issues.

Focus areas:
- O(n²) or worse algorithms where better alternatives exist
- Unnecessary database queries or N+1 problems
- Memory leaks or excessive memory usage
- Blocking operations in async contexts
- Missing caching opportunities
- Inefficient data structures

Return ONLY JSON. No markdown, no commentary.
Only report issues you are confident about from the diff.""",
    ),
    Persona(
        id="maintainability",
        name="Maintainability Reviewer",
        focus="code quality and maintainability",
        system_prompt="""You are a maintainability-focused code reviewer.
Your job is to find code quality issues that make the code harder to maintain.

Focus areas:
- Overly complex functions (high cyclomatic complexity)
- Poor naming or unclear intent
- Missing error handling
- Code duplication that should be abstracted
- Violations of SOLID principles
- Tight coupling between components

Return ONLY JSON. No markdown, no commentary.
Only report issues you are confident about from the diff.""",
    ),
]


class MultiPersonaPipeline(ReviewPipeline):
    """
    G3-multipersona: 여러 페르소나(관점)로 리뷰 후 결과 병합.

    핵심 기능:
    - Security, Performance, Maintainability 등 여러 관점의 리뷰어 시뮬레이션
    - 각 페르소나가 병렬로 리뷰 수행
    - 결과를 병합하여 다각도 리뷰 제공
    """

    def get_personas(self) -> List[Persona]:
        """
        사용할 페르소나 목록 반환.
        params에서 커스텀 페르소나 설정 가능.
        """
        persona_ids = self.params.get("personas", ["security", "performance", "maintainability"])

        # 기본 페르소나에서 선택
        persona_map = {p.id: p for p in DEFAULT_PERSONAS}
        return [persona_map[pid] for pid in persona_ids if pid in persona_map]

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
        Template Method의 run()을 오버라이드하여 multi-persona 로직 구현.
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

        # 3) 페르소나별 병렬 리뷰
        personas = self.get_personas()
        max_concurrency = int(self.params.get("max_concurrency", 3))

        results = await self._review_with_personas(
            req=req,
            diff=diff,
            diff_target=diff_target,
            personas=personas,
            llm=llm,
            format_instructions=format_instructions,
            max_concurrency=max_concurrency,
        )

        # 4) 결과 병합
        merged = await self.reduce_persona_results(results, personas)

        # 5) meta inject
        merged.meta.variant_id = getattr(req, "variant_id", None) or ""
        merged.meta.run_id = run_id
        merged.meta.llm_provider = adapter.provider
        merged.meta.model = adapter.model_name
        merged.meta.diff_target = diff_target
        merged.meta.generated_at = datetime.now().isoformat()

        # 6) 후처리
        await self.after_run(
            req=req,
            result=merged,
            raw_text="",
            raw_json=None,
            fixed_json=None,
        )

        return merged

    async def _review_with_personas(
        self,
        *,
        req,
        diff: str,
        diff_target: str,
        personas: List[Persona],
        llm: AdapterChatModel,
        format_instructions: str,
        max_concurrency: int,
    ) -> List[tuple[Persona, ReviewResult]]:
        """각 페르소나로 병렬 리뷰 수행."""
        semaphore = asyncio.Semaphore(max_concurrency)
        bad_max_chars = int(self.params.get("bad_max_chars", self.pack.params.get("bad_max_chars", 4000)))

        async def review_as_persona(persona: Persona) -> tuple[Persona, ReviewResult]:
            async with semaphore:
                # 페르소나별 프롬프트 구성
                prompt = ChatPromptTemplate.from_messages([
                    ("system", persona.system_prompt),
                    ("human", self.pack.review_user),
                ])
                chain = prompt.partial(format_instructions=format_instructions) | llm

                # repair chain (공통)
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
                payload["persona_name"] = persona.name
                payload["persona_focus"] = persona.focus

                msg = await invoke_chain(chain, payload)
                content = msg.content or ""

                result, _, _, _ = await validate_or_repair(
                    raw_text=content,
                    repair_chain=repair_chain,
                    bad_max_chars=bad_max_chars,
                )

                return (persona, result)

        tasks = [review_as_persona(p) for p in personas]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 성공한 결과만 필터링
        valid_results = [
            r for r in raw_results
            if isinstance(r, tuple) and isinstance(r[1], ReviewResult)
        ]

        return valid_results

    async def reduce_persona_results(
        self,
        results: List[tuple[Persona, ReviewResult]],
        personas: List[Persona],
    ) -> ReviewResult:
        """
        페르소나별 결과를 병합.
        이슈에 페르소나 태그를 추가하여 어떤 관점에서 발견했는지 표시.
        """
        if not results:
            return ReviewResult()

        all_issues: List[Issue] = []
        all_test_suggestions: List[TestSuggestion] = []
        all_questions: List[Question] = []
        all_blockers: List[str] = []
        all_patches: List[PatchSuggestion] = []
        all_key_points: List[str] = []

        persona_summaries: List[str] = []

        for persona, result in results:
            # 이슈에 페르소나 태그 추가
            for issue in result.issues:
                # 카테고리에 페르소나 정보 추가
                original_cat = issue.category or ""
                issue.category = f"[{persona.name}] {original_cat}".strip()
                all_issues.append(issue)

            all_test_suggestions.extend(result.test_suggestions)
            all_questions.extend(result.questions_to_author)
            all_blockers.extend(result.merge_blockers)
            all_patches.extend(result.patch_suggestions)

            # 각 페르소나의 key_points 수집
            for kp in result.summary.key_points:
                all_key_points.append(f"[{persona.id}] {kp}")

            # 페르소나별 요약
            if result.summary.intent:
                persona_summaries.append(f"{persona.name}: {result.summary.intent}")

        # 이슈 ID 재할당
        for i, issue in enumerate(all_issues, 1):
            issue.id = f"ISS-{i:03d}"

        # overall_risk: 가장 높은 위험도 사용
        risk_priority = {RiskLevel.high: 3, RiskLevel.medium: 2, RiskLevel.low: 1}
        max_risk = RiskLevel.low
        for _, result in results:
            if risk_priority.get(result.summary.overall_risk, 0) > risk_priority.get(max_risk, 0):
                max_risk = result.summary.overall_risk

        # 통합 intent 생성
        persona_names = [p.name for p in personas]
        intent = f"Multi-perspective review by {', '.join(persona_names)}"

        merged = ReviewResult(
            meta=Meta(),
            summary=Summary(
                intent=intent,
                overall_risk=max_risk,
                key_points=all_key_points[:15],  # 페르소나별로 합쳐지므로 더 많이
            ),
            issues=all_issues,
            test_suggestions=all_test_suggestions,
            questions_to_author=all_questions,
            merge_blockers=list(set(all_blockers)),
            patch_suggestions=all_patches,
        )

        return merged
