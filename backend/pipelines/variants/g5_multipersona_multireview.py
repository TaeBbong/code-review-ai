"""
G5-MultiPersona-MultiReview Pipeline

G3(Multi-Persona) + G4(Multi-Review) 조합:
- 각 페르소나(correctness, security, performance, maintainability)가 N회 독립 리뷰
- 페르소나별 few-shot 예제로 더 정확한 이슈 탐지
- LLM 기반 집계로 중복 제거 및 신뢰도 평가

핵심 원리:
- 페르소나별 전문성 + 다회 리뷰의 시너지
- Few-shot 예제로 7B 모델의 정확도 향상
- 각 페르소나가 자신의 영역에서 더 깊은 리뷰
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
class PersonaConfig:
    """페르소나 설정 (few-shot 예제 포함)."""
    id: str
    name: str
    focus: str
    system_prompt: str
    few_shot_examples: str  # 각 카테고리별 2개씩 예제


# 각 페르소나별 few-shot 예제 포함 시스템 프롬프트
PERSONA_CONFIGS = {
    "correctness": PersonaConfig(
        id="correctness",
        name="Correctness Reviewer",
        focus="bugs, logic errors, and runtime issues",
        system_prompt="""You are a correctness-focused code reviewer.
Your job is to find bugs, logic errors, and runtime issues that could cause incorrect behavior.

## Focus Areas
- Null/None pointer dereferences without checks
- Off-by-one errors and array index out of bounds
- Type mismatches and incorrect type handling
- Race conditions and concurrency bugs
- Incorrect boolean logic or control flow
- Unhandled edge cases (empty lists, zero values, etc.)
- Resource leaks (unclosed files, connections)

## Output Rules
- Use ONLY these category values: correctness, security, performance, maintainability
- Use ONLY these severity values: high, medium, low
- Return ONLY valid JSON. No markdown, no commentary.
- Only report issues you are confident about from the diff.

## Examples

### Example 1: Off-by-one error
Input diff:
```
--- a/utils.py
+++ b/utils.py
@@ -1,5 +1,5 @@
 def sum_range(start, end):
-    total = 0
-    for i in range(start, end + 1):
-        total += i
-    return total
+    total = 0
+    for i in range(start, end):
+        total += i
+    return total
```

Output:
```json
{{
  "issues": [
    {{
      "id": "ISS-001",
      "title": "Off-by-one error: end value excluded from sum",
      "description": "Changed 'range(start, end + 1)' to 'range(start, end)' excludes the end value from the sum. This breaks the inclusive range behavior.",
      "category": "correctness",
      "severity": "high",
      "locations": [{{"file": "utils.py", "line_start": 3, "line_end": 3}}],
      "suggested_fix": "Use range(start, end + 1) to include the end value"
    }}
  ],
  "summary": {{
    "intent": "Sum range function modification",
    "overall_risk": "high",
    "key_points": ["Critical: end value excluded changes function behavior"]
  }}
}}
```

### Example 2: Missing null check
Input diff:
```
--- a/user_service.py
+++ b/user_service.py
@@ -1,6 +1,5 @@
 def get_user_name(user):
-    if user is not None:
-        return user.name
+    return user.name
```

Output:
```json
{{
  "issues": [
    {{
      "id": "ISS-001",
      "title": "Missing null check causes AttributeError",
      "description": "Removing the null check means calling user.name on None will raise AttributeError. This is a runtime crash waiting to happen.",
      "category": "correctness",
      "severity": "high",
      "locations": [{{"file": "user_service.py", "line_start": 2, "line_end": 2}}],
      "suggested_fix": "Restore the null check: if user is not None: return user.name"
    }}
  ],
  "summary": {{
    "intent": "User name retrieval simplified",
    "overall_risk": "high",
    "key_points": ["Critical: NoneType crash when user is None"]
  }}
}}
```
""",
        few_shot_examples="",  # 포함됨
    ),
    "security": PersonaConfig(
        id="security",
        name="Security Reviewer",
        focus="security vulnerabilities and unsafe patterns",
        system_prompt="""You are a security-focused code reviewer.
Your job is to find security vulnerabilities, potential exploits, and unsafe patterns.

## Focus Areas
- Injection attacks (SQL, command, XSS, etc.)
- Authentication and authorization issues
- Sensitive data exposure
- Input validation problems
- Cryptographic weaknesses
- OWASP Top 10 vulnerabilities

## Output Rules
- Use ONLY these category values: correctness, security, performance, maintainability
- Use ONLY these severity values: high, medium, low
- Return ONLY valid JSON. No markdown, no commentary.
- Only report issues you are confident about from the diff.

## Examples

### Example 1: SQL Injection
Input diff:
```
--- a/user_repo.py
+++ b/user_repo.py
@@ -1,5 +1,5 @@
 def get_user(db, username):
-    query = "SELECT * FROM users WHERE username = ?"
-    return db.execute(query, (username,)).fetchone()
+    query = f"SELECT * FROM users WHERE username = '{{username}}'"
+    return db.execute(query).fetchone()
```

Output:
```json
{{
  "issues": [
    {{
      "id": "ISS-001",
      "title": "SQL Injection vulnerability via string formatting",
      "description": "Using f-string to build SQL query allows SQL injection attacks. An attacker can input malicious SQL like \"'; DROP TABLE users; --\" to compromise the database.",
      "category": "security",
      "severity": "high",
      "locations": [{{"file": "user_repo.py", "line_start": 2, "line_end": 3}}],
      "suggested_fix": "Use parameterized queries: db.execute('SELECT * FROM users WHERE username = ?', (username,))"
    }}
  ],
  "summary": {{
    "intent": "Query building method changed",
    "overall_risk": "high",
    "key_points": ["Critical: SQL injection vulnerability introduced"]
  }}
}}
```

### Example 2: Hardcoded credentials
Input diff:
```
--- a/config.py
+++ b/config.py
@@ -1,4 +1,4 @@
-API_KEY = os.environ.get('API_KEY')
+API_KEY = "sk-1234567890abcdef"
 DATABASE_URL = os.environ.get('DATABASE_URL')
```

Output:
```json
{{
  "issues": [
    {{
      "id": "ISS-001",
      "title": "Hardcoded API key exposed in source code",
      "description": "API key is hardcoded instead of using environment variable. This exposes the secret in version control and could lead to unauthorized access.",
      "category": "security",
      "severity": "high",
      "locations": [{{"file": "config.py", "line_start": 1, "line_end": 1}}],
      "suggested_fix": "Use environment variable: os.environ.get('API_KEY')"
    }}
  ],
  "summary": {{
    "intent": "API key configuration changed",
    "overall_risk": "high",
    "key_points": ["Critical: Secret exposed in source code"]
  }}
}}
```
""",
        few_shot_examples="",
    ),
    "performance": PersonaConfig(
        id="performance",
        name="Performance Reviewer",
        focus="performance bottlenecks and inefficiencies",
        system_prompt="""You are a performance-focused code reviewer.
Your job is to find performance bottlenecks, inefficiencies, and scalability issues.

## Focus Areas
- O(n²) or worse algorithms where better alternatives exist
- Unnecessary database queries or N+1 problems
- Memory leaks or excessive memory usage
- Blocking operations in async contexts
- Missing caching opportunities
- Inefficient data structures

## Output Rules
- Use ONLY these category values: correctness, security, performance, maintainability
- Use ONLY these severity values: high, medium, low
- Return ONLY valid JSON. No markdown, no commentary.
- Only report issues you are confident about from the diff.

## Examples

### Example 1: N+1 Query Problem
Input diff:
```
--- a/user_service.py
+++ b/user_service.py
@@ -1,6 +1,7 @@
 def get_users_with_orders():
-    users = User.query.options(joinedload(User.orders)).all()
+    users = User.query.all()
     return [
-        {{"name": u.name, "order_count": len(u.orders)}}
+        {{"name": u.name, "order_count": len(u.orders)}}  # Triggers query per user
         for u in users
     ]
```

Output:
```json
{{
  "issues": [
    {{
      "id": "ISS-001",
      "title": "N+1 query problem: removed eager loading",
      "description": "Removing joinedload causes a separate database query for each user's orders. With 1000 users, this becomes 1001 queries instead of 1.",
      "category": "performance",
      "severity": "high",
      "locations": [{{"file": "user_service.py", "line_start": 2, "line_end": 4}}],
      "suggested_fix": "Restore eager loading: User.query.options(joinedload(User.orders)).all()"
    }}
  ],
  "summary": {{
    "intent": "User query simplified",
    "overall_risk": "high",
    "key_points": ["Critical: N+1 queries will cause severe performance degradation"]
  }}
}}
```

### Example 2: Quadratic algorithm
Input diff:
```
--- a/search.py
+++ b/search.py
@@ -1,5 +1,7 @@
 def find_common(list1, list2):
-    set2 = set(list2)
-    return [x for x in list1 if x in set2]
+    result = []
+    for x in list1:
+        if x in list2:  # O(n) lookup in list
+            result.append(x)
+    return result
```

Output:
```json
{{
  "issues": [
    {{
      "id": "ISS-001",
      "title": "O(n²) complexity: list lookup instead of set",
      "description": "Changed O(1) set lookup to O(n) list lookup. With 10,000 items in each list, this goes from 10,000 operations to 100,000,000.",
      "category": "performance",
      "severity": "medium",
      "locations": [{{"file": "search.py", "line_start": 4, "line_end": 4}}],
      "suggested_fix": "Convert list2 to set first: set2 = set(list2); [x for x in list1 if x in set2]"
    }}
  ],
  "summary": {{
    "intent": "Common element finding rewritten",
    "overall_risk": "medium",
    "key_points": ["Performance: quadratic complexity for large inputs"]
  }}
}}
```
""",
        few_shot_examples="",
    ),
    "maintainability": PersonaConfig(
        id="maintainability",
        name="Maintainability Reviewer",
        focus="code quality and maintainability issues",
        system_prompt="""You are a maintainability-focused code reviewer.
Your job is to find code quality issues that make the code harder to maintain.

## Focus Areas
- Overly complex functions (high cyclomatic complexity)
- Poor naming or unclear intent
- Missing error handling
- Code duplication that should be abstracted
- Violations of SOLID principles
- Tight coupling between components

## Output Rules
- Use ONLY these category values: correctness, security, performance, maintainability
- Use ONLY these severity values: high, medium, low
- Return ONLY valid JSON. No markdown, no commentary.
- Only report issues you are confident about from the diff.

## Examples

### Example 1: Poor naming and magic numbers
Input diff:
```
--- a/calculator.py
+++ b/calculator.py
@@ -1,5 +1,5 @@
-def calculate_total_price(items, tax_rate):
-    subtotal = sum(item.price for item in items)
-    tax = subtotal * tax_rate
-    return subtotal + tax
+def calc(x, y):
+    a = sum(i.price for i in x)
+    b = a * y
+    return a + b
```

Output:
```json
{{
  "issues": [
    {{
      "id": "ISS-001",
      "title": "Poor naming makes code unreadable",
      "description": "Renamed descriptive names to single letters: calculate_total_price->calc, items->x, tax_rate->y, subtotal->a, tax->b. This destroys code readability.",
      "category": "maintainability",
      "severity": "medium",
      "locations": [{{"file": "calculator.py", "line_start": 1, "line_end": 5}}],
      "suggested_fix": "Use descriptive names: calculate_total_price(items, tax_rate) with subtotal and tax variables"
    }}
  ],
  "summary": {{
    "intent": "Calculator function renamed",
    "overall_risk": "medium",
    "key_points": ["Maintainability: single-letter names destroy readability"]
  }}
}}
```

### Example 2: Deep nesting
Input diff:
```
--- a/validator.py
+++ b/validator.py
@@ -1,10 +1,15 @@
 def validate_order(order):
-    if not order.items:
-        return False
-    if not order.customer:
-        return False
-    if order.total <= 0:
-        return False
-    return True
+    if order:
+        if order.items:
+            if len(order.items) > 0:
+                if order.customer:
+                    if order.customer.id:
+                        if order.total:
+                            if order.total > 0:
+                                return True
+    return False
```

Output:
```json
{{
  "issues": [
    {{
      "id": "ISS-001",
      "title": "Deep nesting reduces readability (arrow anti-pattern)",
      "description": "7 levels of nested if statements (arrow anti-pattern). Each nesting level increases cognitive complexity. The original early-return pattern was cleaner.",
      "category": "maintainability",
      "severity": "medium",
      "locations": [{{"file": "validator.py", "line_start": 2, "line_end": 9}}],
      "suggested_fix": "Use early returns: if not order: return False; if not order.items: return False; etc."
    }}
  ],
  "summary": {{
    "intent": "Order validation rewritten",
    "overall_risk": "medium",
    "key_points": ["Maintainability: arrow anti-pattern makes logic hard to follow"]
  }}
}}
```
""",
        few_shot_examples="",
    ),
}


class MultiPersonaMultiReviewPipeline(ReviewPipeline):
    """
    G5-multipersona-multireview: 페르소나별 N회 독립 리뷰 후 LLM 집계.

    핵심 기능:
    - 각 페르소나(correctness, security, performance, maintainability)가 전문 영역에 집중
    - 각 페르소나가 N회 독립 리뷰 수행 (LLM 무작위성 활용)
    - Few-shot 예제로 더 정확한 이슈 탐지
    - LLM 기반 집계로 중복 제거 및 신뢰도 평가

    Attributes:
        last_persona_results: 마지막 실행의 페르소나별 리뷰 결과 (디버깅용)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_persona_results: Dict[str, List[ReviewResult]] = {}

    def get_personas(self) -> List[PersonaConfig]:
        """사용할 페르소나 목록 반환."""
        persona_ids = self.params.get(
            "personas",
            ["correctness", "security", "performance", "maintainability"]
        )
        return [PERSONA_CONFIGS[pid] for pid in persona_ids if pid in PERSONA_CONFIGS]

    async def resolve_diff(self, req):
        raw = (getattr(req, "diff", None) or "").strip()
        if raw:
            return raw, "raw"

        diff_target = (
            getattr(req, "diff_target", None)
            or self.params.get("diff_source")
            or "staged"
        ).strip()
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
        Template Method의 run()을 오버라이드하여 multipersona-multireview 로직 구현.
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
        bad_max_chars = int(
            self.params.get("bad_max_chars", self.pack.params.get("bad_max_chars", 4000))
        )

        # 3) 페르소나별 N회 리뷰 수행
        personas = self.get_personas()
        num_reviews_per_persona = int(self.params.get("num_reviews_per_persona", 2))
        max_concurrency = int(self.params.get("max_concurrency", 4))

        persona_results = await self._run_persona_reviews(
            req=req,
            diff=diff,
            diff_target=diff_target,
            personas=personas,
            llm=llm,
            format_instructions=format_instructions,
            bad_max_chars=bad_max_chars,
            num_reviews_per_persona=num_reviews_per_persona,
            max_concurrency=max_concurrency,
        )

        # 페르소나별 결과 저장 (디버깅용)
        self.last_persona_results = persona_results

        # 4) 결과 집계
        all_results: List[ReviewResult] = []
        for results in persona_results.values():
            all_results.extend(results)

        aggregation_mode = self.params.get("aggregation_mode", "llm")

        if aggregation_mode == "llm" and len(all_results) > 1:
            aggregated = await self._aggregate_with_llm(
                review_results=all_results,
                persona_results=persona_results,
                diff=diff,
                llm=llm,
                format_instructions=format_instructions,
                bad_max_chars=bad_max_chars,
            )
        else:
            aggregated = self._aggregate_simple(all_results, persona_results)

        # 5) meta inject
        aggregated.meta.variant_id = getattr(req, "variant_id", None) or ""
        aggregated.meta.run_id = run_id
        aggregated.meta.llm_provider = adapter.provider
        aggregated.meta.model = adapter.model_name
        aggregated.meta.diff_target = diff_target
        aggregated.meta.generated_at = datetime.now().isoformat()

        # 6) 통계 정보 추가
        total_raw_issues = sum(len(r.issues) for r in all_results)
        persona_summary = ", ".join(
            f"{p.id}:{len(persona_results.get(p.id, []))*num_reviews_per_persona}"
            for p in personas
        )
        aggregated.summary.key_points.insert(
            0,
            f"Aggregated from {len(personas)} personas × {num_reviews_per_persona} reviews "
            f"({total_raw_issues} raw issues → {len(aggregated.issues)} final)"
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

    async def _run_persona_reviews(
        self,
        *,
        req,
        diff: str,
        diff_target: str,
        personas: List[PersonaConfig],
        llm: AdapterChatModel,
        format_instructions: str,
        bad_max_chars: int,
        num_reviews_per_persona: int,
        max_concurrency: int,
    ) -> Dict[str, List[ReviewResult]]:
        """각 페르소나별 N회 독립 리뷰 수행."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def single_review(
            persona: PersonaConfig,
            review_idx: int
        ) -> tuple[str, ReviewResult]:
            async with semaphore:
                # 페르소나별 프롬프트 구성 (few-shot 포함)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", persona.system_prompt),
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
                payload["persona_name"] = persona.name
                payload["persona_focus"] = persona.focus
                payload["review_pass"] = review_idx + 1

                msg = await invoke_chain(chain, payload)
                content = msg.content or ""

                result, _, _, _ = await validate_or_repair(
                    raw_text=content,
                    repair_chain=repair_chain,
                    bad_max_chars=bad_max_chars,
                )

                # 이슈에 페르소나 및 리뷰 패스 정보 추가
                for issue in result.issues:
                    original_title = issue.title or ""
                    issue.title = f"[{persona.name}] {original_title}".strip()
                    if not issue.evidence_ids:
                        issue.evidence_ids = []
                    issue.evidence_ids.append(f"{persona.id}_pass_{review_idx + 1}")

                return (persona.id, result)

        # 모든 페르소나 × N회 리뷰 태스크 생성
        tasks = [
            single_review(persona, review_idx)
            for persona in personas
            for review_idx in range(num_reviews_per_persona)
        ]

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과를 페르소나별로 그룹화
        persona_results: Dict[str, List[ReviewResult]] = {p.id: [] for p in personas}
        for r in raw_results:
            if isinstance(r, tuple) and isinstance(r[1], ReviewResult):
                persona_id, result = r
                persona_results[persona_id].append(result)

        return persona_results

    async def _aggregate_with_llm(
        self,
        *,
        review_results: List[ReviewResult],
        persona_results: Dict[str, List[ReviewResult]],
        diff: str,
        llm: AdapterChatModel,
        format_instructions: str,
        bad_max_chars: int,
    ) -> ReviewResult:
        """
        LLM 기반 집계: 페르소나별 결과를 분석하여 중복 제거 및 신뢰도 평가.
        """
        all_issues_text = self._format_all_issues_for_aggregation(
            review_results, persona_results
        )

        aggregation_system = """You are a code review aggregator.
You have received multiple reviews from different specialized reviewers (personas).
Each persona has reviewed the code multiple times independently.

Your job is to merge these reviews into a single, high-quality review report.

Guidelines:
1. MERGE duplicate issues: Same issue found by multiple reviewers/passes should be combined.
2. INCREASE confidence for repeated issues: Issues found by multiple personas or passes are more reliable.
3. KEEP unique issues: Even single-occurrence issues may be valid if the reasoning is sound.
4. REMOVE obvious false positives: Only remove issues that are clearly wrong.
5. PRIORITIZE: correctness > security > performance > maintainability

Return ONLY JSON. No markdown, no commentary."""

        aggregation_user = """Aggregate the following reviews from multiple specialized reviewers.

{format_instructions}

## Original Diff
{diff}

## Reviews by Persona
{all_issues_text}

Instructions:
- Merge semantically duplicate issues (same bug, different wording)
- Note how many reviewers/passes found each issue (e.g., "Found by 3/8 reviews")
- Assign higher confidence to issues found by multiple personas
- Keep valid unique issues even if only one reviewer found them
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

    def _aggregate_simple(
        self,
        all_results: List[ReviewResult],
        persona_results: Dict[str, List[ReviewResult]],
    ) -> ReviewResult:
        """단순 합산 집계 (fallback)."""
        if not all_results:
            return ReviewResult()

        all_issues: List[Issue] = []
        all_test_suggestions: List[TestSuggestion] = []
        all_questions: List[Question] = []
        all_blockers: List[str] = []
        all_patches: List[PatchSuggestion] = []
        all_key_points: List[str] = []

        for result in all_results:
            all_issues.extend(result.issues)
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
        for result in all_results:
            if risk_priority.get(result.summary.overall_risk, 0) > risk_priority.get(max_risk, 0):
                max_risk = result.summary.overall_risk

        merged = ReviewResult(
            meta=Meta(),
            summary=Summary(
                intent="Multi-persona multi-review aggregation",
                overall_risk=max_risk,
                key_points=all_key_points[:15],
            ),
            issues=all_issues,
            test_suggestions=all_test_suggestions,
            questions_to_author=all_questions,
            merge_blockers=list(set(all_blockers)),
            patch_suggestions=all_patches,
        )

        return merged

    def _format_all_issues_for_aggregation(
        self,
        review_results: List[ReviewResult],
        persona_results: Dict[str, List[ReviewResult]],
    ) -> str:
        """페르소나별 리뷰 결과를 집계용 텍스트로 변환."""
        sections = []

        for persona_id, results in persona_results.items():
            if not results:
                continue

            persona_config = PERSONA_CONFIGS.get(persona_id)
            persona_name = persona_config.name if persona_config else persona_id

            section_lines = [f"### {persona_name} ({len(results)} reviews)"]

            for pass_idx, result in enumerate(results, 1):
                section_lines.append(f"\n#### Pass {pass_idx}")
                section_lines.append(f"Risk: {result.summary.overall_risk.value if result.summary.overall_risk else 'unknown'}")
                section_lines.append(f"Issues: {len(result.issues)}")

                for issue in result.issues:
                    locations = ", ".join(
                        f"{loc.file}:{loc.line_start}-{loc.line_end}"
                        for loc in (issue.locations or [])
                    ) or "unknown"

                    section_lines.append(f"""
Issue: {issue.title}
Severity: {issue.severity.value if issue.severity else 'unknown'}
Category: {issue.category.value if issue.category else 'unknown'}
Location: {locations}
Description: {issue.description}
""")

            sections.append("\n".join(section_lines))

        return "\n\n---\n\n".join(sections)
