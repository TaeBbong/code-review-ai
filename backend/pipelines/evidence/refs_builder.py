from __future__ import annotations

from typing import Any, Dict, List

from backend.domain.tools.get_symbols import ChangedSymbol, extract_changed_symbols
from backend.domain.tools.ripgrep_refs import search_symbol_references
from backend.domain.tools.similar_code import find_similar_to_diff
from backend.domain.tools.symbol_definition import get_definitions_for_symbols


def _make_ref_id(i: int) -> str:
    return f"E-REF-{i:03d}"


def _make_def_id(i: int) -> str:
    return f"E-DEF-{i:03d}"


def _make_sim_id(i: int) -> str:
    return f"E-SIM-{i:03d}"


async def build_refs_evidence(
    *,
    diff_text: str,
    repo_root: str,
    max_symbols: int = 12,
    top_k_per_symbol: int = 6,
) -> List[Dict[str, Any]]:
    """
    evidence_pack.refs 채우는 v0.
    출력 각 item은 최소한 다음을 가짐:
    - id (E-REF-xxx)
    - symbol, kind, file_path(language), hit(path/line/text), score, reason
    """
    symbols: list[ChangedSymbol] = extract_changed_symbols(diff_text, max_symbols=max_symbols)

    refs: list[Dict[str, Any]] = []
    eid = 1

    for sym in symbols:
        hits = await search_symbol_references(
            repo_root=repo_root,
            symbol=sym.name,
            changed_file=sym.file_path,
            top_k=top_k_per_symbol,
        )

        for h in hits:
            refs.append(
                {
                    "id": _make_ref_id(eid),
                    "type": "ref",
                    "symbol": {
                        "name": sym.name,
                        "kind": sym.kind,
                        "language": sym.language,
                        "changed_file": sym.file_path,
                        "hint": sym.evidence_hint,
                    },
                    "hit": {
                        "path": h.path,
                        "line": h.line,
                        "text": h.text,
                    },
                    "rank": {
                        "score": h.score,
                        "reason": h.reason,
                    },
                }
            )
            eid += 1

    return refs


async def build_definitions_evidence(
    *,
    diff_text: str,
    repo_root: str,
    max_definitions: int = 8,
    max_body_lines: int = 40,
) -> List[Dict[str, Any]]:
    """
    diff에서 참조하는 외부 심볼의 정의 본문을 수집.
    새 코드가 호출하는 함수/클래스의 실제 구현을 LLM에게 제공.

    출력 item:
    - id (E-DEF-xxx)
    - type: "definition"
    - symbol: 심볼 이름, 종류
    - location: 파일 경로, 라인 범위
    - body: 정의 본문
    """
    symbols: list[ChangedSymbol] = extract_changed_symbols(diff_text, max_symbols=max_definitions * 2)
    symbol_names = [s.name for s in symbols]

    definitions = await get_definitions_for_symbols(
        repo_root=repo_root,
        symbols=symbol_names,
        max_definitions=max_definitions,
        max_body_lines=max_body_lines,
    )

    defs: list[Dict[str, Any]] = []
    for i, d in enumerate(definitions, 1):
        defs.append(
            {
                "id": _make_def_id(i),
                "type": "definition",
                "symbol": {
                    "name": d.symbol,
                    "kind": d.kind,
                },
                "location": {
                    "path": d.file_path,
                    "start_line": d.start_line,
                    "end_line": d.end_line,
                },
                "signature": d.signature,
                "body": d.body,
            }
        )

    return defs


async def build_similar_code_evidence(
    *,
    diff_text: str,
    repo_root: str,
    min_similarity: float = 0.3,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    diff의 새 코드와 유사한 기존 코드를 찾아서 반환.
    중복 코드나 일관성 없는 패턴 발견에 유용.

    출력 item:
    - id (E-SIM-xxx)
    - type: "similar"
    - location: 파일 경로, 라인 범위
    - similarity: 유사도 (0.0 ~ 1.0)
    - snippet: 유사 코드 스니펫
    - reason: 매칭 이유
    """
    matches = await find_similar_to_diff(
        repo_root=repo_root,
        diff_text=diff_text,
        min_similarity=min_similarity,
        max_results=max_results,
    )

    similar: list[Dict[str, Any]] = []
    for i, m in enumerate(matches, 1):
        similar.append(
            {
                "id": _make_sim_id(i),
                "type": "similar",
                "location": {
                    "path": m.file_path,
                    "start_line": m.start_line,
                    "end_line": m.end_line,
                },
                "similarity": round(m.similarity, 3),
                "snippet": m.snippet,
                "reason": m.reason,
            }
        )

    return similar


async def build_full_evidence(
    *,
    diff_text: str,
    repo_root: str,
    include_refs: bool = True,
    include_definitions: bool = True,
    include_similar: bool = True,
    max_symbols: int = 12,
    top_k_per_symbol: int = 6,
    max_definitions: int = 8,
    max_body_lines: int = 40,
    min_similarity: float = 0.3,
    max_similar_results: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    전체 evidence를 수집하는 통합 함수.
    필요한 evidence 종류를 선택적으로 포함 가능.

    Returns:
        {
            "refs": [...],       # 심볼 사용처
            "definitions": [...], # 심볼 정의 본문
            "similar": [...]     # 유사 코드
        }
    """
    import asyncio

    tasks = []
    task_names = []

    if include_refs:
        tasks.append(build_refs_evidence(
            diff_text=diff_text,
            repo_root=repo_root,
            max_symbols=max_symbols,
            top_k_per_symbol=top_k_per_symbol,
        ))
        task_names.append("refs")

    if include_definitions:
        tasks.append(build_definitions_evidence(
            diff_text=diff_text,
            repo_root=repo_root,
            max_definitions=max_definitions,
            max_body_lines=max_body_lines,
        ))
        task_names.append("definitions")

    if include_similar:
        tasks.append(build_similar_code_evidence(
            diff_text=diff_text,
            repo_root=repo_root,
            min_similarity=min_similarity,
            max_results=max_similar_results,
        ))
        task_names.append("similar")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    evidence: Dict[str, List[Dict[str, Any]]] = {}
    for name, result in zip(task_names, results):
        if isinstance(result, Exception):
            evidence[name] = []  # 에러 시 빈 리스트
        else:
            evidence[name] = result

    return evidence
