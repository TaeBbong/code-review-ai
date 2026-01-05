from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RefHit:
    path: str
    line: int
    text: str
    score: int
    reason: str


def _is_test_path(p: str) -> bool:
    lp = p.lower()
    return (
        "/test/" in lp
        or "/tests/" in lp
        or "__tests__" in lp
        or lp.endswith(".test.ts")
        or lp.endswith(".spec.ts")
        or ".test." in lp
        or ".spec." in lp
    )


def _score_hit(path: str, *, changed_file: Optional[str]) -> tuple[int, str]:
    """
    높은 점수 우선.
    """
    score = 0
    reasons: list[str] = []

    if _is_test_path(path):
        score += 100
        reasons.append("test-file")

    if changed_file and os.path.normpath(path) == os.path.normpath(changed_file):
        score += 50
        reasons.append("same-file")

    # docs/ 는 보통 낮게(원하면 조정)
    if "/docs/" in path.lower():
        score -= 5
        reasons.append("docs")

    return score, ",".join(reasons) if reasons else "default"


async def search_symbol_references(
    *,
    repo_root: str,
    symbol: str,
    changed_file: Optional[str] = None,
    top_k: int = 8,
    timeout_sec: float = 6.0,
) -> list[RefHit]:
    """
    rg로 심볼 레퍼런스 검색.
    - -n: line number
    - --no-heading / --color never: 파싱 안정화
    - -S: smart-case
    - --glob 로 대형 디렉토리 제외
    """
    cmd = [
        "rg",
        "-n",
        "--no-heading",
        "--color",
        "never",
        "-S",
        # 너무 흔한 곳 제외(필요시 추가)
        "--glob",
        "!.git/**",
        "--glob",
        "!node_modules/**",
        "--glob",
        "!dist/**",
        "--glob",
        "!build/**",
        "--glob",
        "!.dart_tool/**",
        "--glob",
        "!coverage/**",
        symbol,
        ".",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=repo_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            proc.kill()
            return []

    except FileNotFoundError:
        # rg가 설치 안된 경우
        return []

    # rg는 매칭 없으면 exit code 1일 수 있음(정상)
    out = stdout.decode("utf-8", errors="replace").splitlines()

    hits: list[RefHit] = []
    for line in out:
        # format: path:line:text (단, path에 ':'가 들어가면 깨질 수 있지만 일반 repo에서는 대부분 안전)
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        path, line_no, text = parts
        try:
            n = int(line_no)
        except ValueError:
            continue

        score, reason = _score_hit(path, changed_file=changed_file)
        hits.append(RefHit(path=path, line=n, text=text.strip(), score=score, reason=reason))

    # score desc, path asc, line asc
    hits.sort(key=lambda h: (-h.score, h.path, h.line))
    return hits[:top_k]
