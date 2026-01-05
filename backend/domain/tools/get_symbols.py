from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class ChangedSymbol:
    name: str
    kind: str               # "function" | "class" | "method" | "type" | ...
    language: str           # "py" | "ts" | "js" | "dart" | "unknown"
    file_path: str          # b/ 기준 path
    evidence_hint: str      # why we think it's a symbol (matched line excerpt)


# 너무 흔해서 레퍼런스 검색이 의미 없거나 노이즈 큰 심볼들(필요하면 늘려도 됨)
_STOP_SYMBOLS = {
    "main", "test", "tests",
    "init", "__init__", "setup", "teardown",
    "build", "dispose", "render",
    "toString", "equals", "hashCode",
    "get", "set",
}


_DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$")
# hunk header: @@ -a,b +c,d @@
_HUNK_RE = re.compile(r"^@@\s+")

# 언어별 간단 매칭(추후 Day 5/6에서 강화 가능)
_PY_DEF_RE = re.compile(r"^\s*(async\s+def|def)\s+([A-Za-z_]\w*)\s*\(")
_PY_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*(\(|:)")
_TS_FUNC_RE = re.compile(r"^\s*(export\s+)?(async\s+)?function\s+([A-Za-z_]\w*)\s*\(")
_TS_CLASS_RE = re.compile(r"^\s*(export\s+)?class\s+([A-Za-z_]\w*)\s*({|extends|\s)")
_TS_CONST_ARROW_RE = re.compile(r"^\s*(export\s+)?(const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(async\s+)?\(")
_TS_CONST_ARROW2_RE = re.compile(r"^\s*(export\s+)?(const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(async\s+)?\w*\s*=>")

_DART_CLASS_RE = re.compile(r"^\s*(abstract\s+)?class\s+([A-Za-z_]\w*)\s*({|extends|\s)")
_DART_ENUM_RE = re.compile(r"^\s*enum\s+([A-Za-z_]\w*)\s*({|\s)")
_DART_EXT_RE = re.compile(r"^\s*extension\s+([A-Za-z_]\w*)\s+on\s+")
_DART_TOPLEVEL_FUNC_RE = re.compile(
    r"^\s*(?:Future<.*?>|Future|Stream<.*?>|Stream|void|int|double|num|bool|String|Widget|dynamic|[\w<>,\s\?\[\]]+)\s+([A-Za-z_]\w*)\s*\("
)


def _guess_lang(file_path: str) -> str:
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


def _iter_changed_lines(diff_text: str) -> Iterator[tuple[str, str]]:
    """
    yields: (file_path, line_text_without_prefix)
    Only considers +/- lines (excluding '+++'/'---').
    """
    current_file: Optional[str] = None
    for raw in diff_text.splitlines():
        m = _DIFF_FILE_RE.match(raw)
        if m:
            current_file = m.group(2)  # b/ path
            continue
        if current_file is None:
            continue

        # skip hunk headers and file headers
        if _HUNK_RE.match(raw) or raw.startswith("+++ ") or raw.startswith("--- "):
            continue

        if raw.startswith("+") or raw.startswith("-"):
            # avoid counting "+++" or "---" already handled above
            yield current_file, raw[1:]


def extract_changed_symbols(diff_text: str, *, max_symbols: int = 12) -> list[ChangedSymbol]:
    """
    Regex 기반 v0 심볼 추출.
    - diff의 +/- 라인에서 def/class/function/const arrow/dart top-level 등 잡기
    - 중복 제거 + stop list 제거
    - max_symbols 로 상한
    """
    seen: set[tuple[str, str]] = set()  # (file_path, symbol)
    out: list[ChangedSymbol] = []

    for file_path, line in _iter_changed_lines(diff_text):
        lang = _guess_lang(file_path)
        sym = _match_symbol(lang, line)
        if sym is None:
            continue

        name, kind, hint = sym
        if name in _STOP_SYMBOLS:
            continue

        key = (file_path, name)
        if key in seen:
            continue
        seen.add(key)

        out.append(
            ChangedSymbol(
                name=name,
                kind=kind,
                language=lang,
                file_path=file_path,
                evidence_hint=hint,
            )
        )

        if len(out) >= max_symbols:
            break

    return out


def _match_symbol(lang: str, line: str) -> Optional[tuple[str, str, str]]:
    """
    returns (symbol_name, kind, hint_excerpt) or None
    """
    hint = line.strip()[:180]

    if lang == "py":
        m = _PY_DEF_RE.match(line)
        if m:
            return (m.group(2), "function", hint)
        m = _PY_CLASS_RE.match(line)
        if m:
            return (m.group(1), "class", hint)
        return None

    if lang in ("ts", "js"):
        m = _TS_FUNC_RE.match(line)
        if m:
            return (m.group(3), "function", hint)
        m = _TS_CLASS_RE.match(line)
        if m:
            return (m.group(2), "class", hint)
        m = _TS_CONST_ARROW_RE.match(line)
        if m:
            return (m.group(3), "function", hint)
        m = _TS_CONST_ARROW2_RE.match(line)
        if m:
            return (m.group(3), "function", hint)
        return None

    if lang == "dart":
        m = _DART_CLASS_RE.match(line)
        if m:
            return (m.group(2), "class", hint)
        m = _DART_ENUM_RE.match(line)
        if m:
            return (m.group(1), "type", hint)
        m = _DART_EXT_RE.match(line)
        if m:
            return (m.group(1), "type", hint)
        m = _DART_TOPLEVEL_FUNC_RE.match(line)
        if m:
            return (m.group(1), "function", hint)
        return None

    # unknown: 시도는 해볼 수 있지만 v0에서는 과감히 패스
    return None
