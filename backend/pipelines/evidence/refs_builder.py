from __future__ import annotations

from typing import Any, Dict, List

from backend.domain.tools.get_symbols import ChangedSymbol, extract_changed_symbols
from backend.domain.tools.ripgrep_refs import search_symbol_references


def _make_ref_id(i: int) -> str:
    return f"E-REF-{i:03d}"


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
