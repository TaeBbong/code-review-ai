"""
symbol_definition.py

diff에서 참조하는 외부 심볼의 정의(함수/클래스 본문)를 가져오는 도구.
ripgrep으로 정의 위치를 찾고, 해당 파일에서 본문을 추출.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SymbolDefinition:
    """심볼 정의 정보"""
    symbol: str
    kind: str  # "function" | "class" | "method"
    file_path: str
    start_line: int
    end_line: int
    body: str  # 정의 본문
    signature: str  # 첫 줄 (시그니처)


# 언어별 정의 패턴
_DEFINITION_PATTERNS = {
    "py": [
        # def function_name(
        (r"^\s*(async\s+)?def\s+{symbol}\s*\(", "function"),
        # class ClassName
        (r"^\s*class\s+{symbol}\s*[\(:]", "class"),
    ],
    "ts": [
        # function functionName(
        (r"^\s*(export\s+)?(async\s+)?function\s+{symbol}\s*[\(<]", "function"),
        # class ClassName
        (r"^\s*(export\s+)?class\s+{symbol}\s*[{<]", "class"),
        # const funcName = ( or const funcName = async (
        (r"^\s*(export\s+)?(const|let)\s+{symbol}\s*=\s*(async\s+)?[\(<]", "function"),
        # interface InterfaceName
        (r"^\s*(export\s+)?interface\s+{symbol}\s*[{<]", "interface"),
        # type TypeName =
        (r"^\s*(export\s+)?type\s+{symbol}\s*[=<]", "type"),
    ],
    "js": [
        (r"^\s*(export\s+)?(async\s+)?function\s+{symbol}\s*\(", "function"),
        (r"^\s*(export\s+)?class\s+{symbol}\s*[{]", "class"),
        (r"^\s*(export\s+)?(const|let|var)\s+{symbol}\s*=\s*(async\s+)?\(", "function"),
    ],
    "dart": [
        (r"^\s*(abstract\s+)?class\s+{symbol}\s*[{<]", "class"),
        (r"^\s*enum\s+{symbol}\s*[{]", "enum"),
        # 일반 함수 (반환타입 symbol(
        (r"^\s*[\w<>,\?\[\]\s]+\s+{symbol}\s*\(", "function"),
    ],
}


def _guess_lang(file_path: str) -> str:
    """파일 확장자로 언어 추측"""
    fp = file_path.lower()
    if fp.endswith(".py"):
        return "py"
    if fp.endswith(".ts") or fp.endswith(".tsx"):
        return "ts"
    if fp.endswith(".js") or fp.endswith(".jsx"):
        return "js"
    if fp.endswith(".dart"):
        return "dart"
    return "unknown"


def _find_definition_end(lines: list[str], start_idx: int, lang: str) -> int:
    """
    정의의 끝 라인을 찾음.
    간단한 휴리스틱: 들여쓰기 기반 (Python) 또는 중괄호 매칭 (그 외)
    """
    if start_idx >= len(lines):
        return start_idx

    start_line = lines[start_idx]

    if lang == "py":
        # Python: 들여쓰기 기반
        # 시작 라인의 들여쓰기 레벨 확인
        base_indent = len(start_line) - len(start_line.lstrip())

        for i in range(start_idx + 1, min(start_idx + 200, len(lines))):
            line = lines[i]
            stripped = line.strip()

            # 빈 줄이나 주석은 건너뜀
            if not stripped or stripped.startswith("#"):
                continue

            current_indent = len(line) - len(line.lstrip())

            # 같은 레벨이거나 더 낮은 들여쓰기면 끝
            if current_indent <= base_indent:
                return i - 1

        return min(start_idx + 50, len(lines) - 1)

    else:
        # 중괄호 매칭 (JS/TS/Dart)
        brace_count = 0
        found_open = False

        for i in range(start_idx, min(start_idx + 300, len(lines))):
            line = lines[i]

            for char in line:
                if char == "{":
                    brace_count += 1
                    found_open = True
                elif char == "}":
                    brace_count -= 1
                    if found_open and brace_count == 0:
                        return i

        # 중괄호를 못 찾으면 적당히 끊음
        return min(start_idx + 30, len(lines) - 1)


async def find_symbol_definition(
    *,
    repo_root: str,
    symbol: str,
    exclude_file: Optional[str] = None,
    timeout_sec: float = 5.0,
    max_body_lines: int = 50,
) -> Optional[SymbolDefinition]:
    """
    심볼의 정의를 찾아서 반환.

    Args:
        repo_root: 검색할 저장소 루트
        symbol: 찾을 심볼 이름
        exclude_file: 제외할 파일 (diff의 파일)
        timeout_sec: ripgrep 타임아웃
        max_body_lines: 최대 본문 라인 수

    Returns:
        SymbolDefinition or None
    """
    # ripgrep으로 정의 위치 후보 검색
    # -n: 라인 번호, -l 대신 직접 파싱
    cmd = [
        "rg",
        "-n",
        "--no-heading",
        "--color", "never",
        "-S",
        "--glob", "!.git/**",
        "--glob", "!node_modules/**",
        "--glob", "!dist/**",
        "--glob", "!build/**",
        "--glob", "!__pycache__/**",
        "--glob", "!*.min.js",
        # 정의 패턴 검색 (대략적)
        rf"(def|class|function|const|let|interface|type|enum)\s+{re.escape(symbol)}\b",
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
            return None
    except FileNotFoundError:
        return None

    results = stdout.decode("utf-8", errors="replace").splitlines()

    # 결과 파싱 및 필터링
    for line in results:
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue

        file_path, line_no_str, content = parts

        # 제외 파일 건너뛰기
        if exclude_file and Path(file_path).resolve() == Path(exclude_file).resolve():
            continue

        try:
            line_no = int(line_no_str)
        except ValueError:
            continue

        lang = _guess_lang(file_path)
        patterns = _DEFINITION_PATTERNS.get(lang, [])

        # 정의 패턴 매칭 확인
        for pattern_template, kind in patterns:
            pattern = pattern_template.format(symbol=re.escape(symbol))
            if re.match(pattern, content):
                # 정의를 찾음 - 본문 추출
                full_path = Path(repo_root) / file_path
                if not full_path.exists():
                    continue

                try:
                    file_lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
                except Exception:
                    continue

                start_idx = line_no - 1  # 0-indexed
                if start_idx >= len(file_lines):
                    continue

                end_idx = _find_definition_end(file_lines, start_idx, lang)

                # max_body_lines 제한
                if end_idx - start_idx > max_body_lines:
                    end_idx = start_idx + max_body_lines

                body_lines = file_lines[start_idx:end_idx + 1]

                return SymbolDefinition(
                    symbol=symbol,
                    kind=kind,
                    file_path=file_path,
                    start_line=line_no,
                    end_line=end_idx + 1,
                    body="\n".join(body_lines),
                    signature=file_lines[start_idx].strip(),
                )

    return None


async def get_definitions_for_symbols(
    *,
    repo_root: str,
    symbols: list[str],
    exclude_file: Optional[str] = None,
    max_definitions: int = 10,
    max_body_lines: int = 50,
) -> list[SymbolDefinition]:
    """
    여러 심볼의 정의를 병렬로 가져옴.

    Args:
        repo_root: 저장소 루트
        symbols: 찾을 심볼 목록
        exclude_file: 제외할 파일
        max_definitions: 최대 정의 개수
        max_body_lines: 정의당 최대 라인 수

    Returns:
        SymbolDefinition 목록
    """
    tasks = [
        find_symbol_definition(
            repo_root=repo_root,
            symbol=sym,
            exclude_file=exclude_file,
            max_body_lines=max_body_lines,
        )
        for sym in symbols[:max_definitions * 2]  # 여유있게 검색
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    definitions = []
    for result in results:
        if isinstance(result, SymbolDefinition):
            definitions.append(result)
            if len(definitions) >= max_definitions:
                break

    return definitions
