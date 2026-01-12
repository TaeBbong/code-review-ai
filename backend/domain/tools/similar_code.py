"""
similar_code.py

n-gram 토큰 유사도 기반 유사 코드 탐지 도구.
새로 작성된 코드와 비슷한 패턴의 기존 코드를 찾아서 반환.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SimilarCodeMatch:
    """유사 코드 매칭 결과"""
    file_path: str
    start_line: int
    end_line: int
    snippet: str  # 매칭된 코드 스니펫
    similarity: float  # 0.0 ~ 1.0
    matched_ngrams: int  # 매칭된 n-gram 수
    total_ngrams: int  # 쿼리의 총 n-gram 수
    reason: str  # 매칭 이유


@dataclass
class TokenizedCode:
    """토큰화된 코드"""
    tokens: list[str]
    ngrams: set[tuple[str, ...]]
    original: str


# 토큰화에서 제외할 키워드 (너무 흔해서 유사도에 노이즈)
_COMMON_KEYWORDS = {
    # Python
    "def", "class", "return", "if", "else", "elif", "for", "while", "import",
    "from", "try", "except", "finally", "with", "as", "pass", "break", "continue",
    "and", "or", "not", "in", "is", "None", "True", "False", "self", "async", "await",
    # JS/TS
    "function", "const", "let", "var", "export", "default", "import", "require",
    "new", "this", "typeof", "instanceof", "null", "undefined", "true", "false",
    # 공통
    "public", "private", "protected", "static", "final", "void", "int", "string",
}

# 토큰 추출 정규식
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize(code: str, filter_common: bool = True) -> list[str]:
    """
    코드를 토큰으로 분리.
    식별자만 추출하고, 너무 흔한 키워드는 필터링.
    """
    tokens = _TOKEN_RE.findall(code)

    if filter_common:
        tokens = [t for t in tokens if t.lower() not in _COMMON_KEYWORDS and len(t) > 1]

    return tokens


def _make_ngrams(tokens: list[str], n: int = 3) -> set[tuple[str, ...]]:
    """토큰 리스트에서 n-gram 집합 생성"""
    if len(tokens) < n:
        # 토큰이 부족하면 있는 대로
        if tokens:
            return {tuple(tokens)}
        return set()

    ngrams = set()
    for i in range(len(tokens) - n + 1):
        ngrams.add(tuple(tokens[i:i + n]))
    return ngrams


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """자카드 유사도 계산"""
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


def _tokenize_code(code: str, ngram_size: int = 3) -> TokenizedCode:
    """코드를 토큰화하고 n-gram 생성"""
    tokens = _tokenize(code)
    ngrams = _make_ngrams(tokens, ngram_size)
    return TokenizedCode(tokens=tokens, ngrams=ngrams, original=code)


def _extract_code_blocks(file_content: str, lang: str) -> list[tuple[int, int, str]]:
    """
    파일에서 코드 블록(함수/클래스/메서드)을 추출.
    Returns: [(start_line, end_line, code_block), ...]
    """
    lines = file_content.splitlines()
    blocks: list[tuple[int, int, str]] = []

    if lang == "py":
        # Python: def/class로 시작하는 블록
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()

            if stripped.startswith(("def ", "async def ", "class ")):
                base_indent = len(line) - len(stripped)
                start = i
                end = i

                # 블록 끝 찾기
                for j in range(i + 1, min(i + 200, len(lines))):
                    next_line = lines[j]
                    next_stripped = next_line.strip()

                    if not next_stripped:
                        continue

                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent <= base_indent and next_stripped:
                        break
                    end = j

                block = "\n".join(lines[start:end + 1])
                blocks.append((start + 1, end + 1, block))
                i = end + 1
            else:
                i += 1

    else:
        # JS/TS/Dart: 중괄호 기반
        i = 0
        while i < len(lines):
            line = lines[i]

            # 함수/클래스 정의 패턴
            if re.search(r"(function\s+\w+|class\s+\w+|const\s+\w+\s*=|=>\s*\{)", line):
                start = i
                brace_count = 0
                found_open = False

                for j in range(i, min(i + 300, len(lines))):
                    for char in lines[j]:
                        if char == "{":
                            brace_count += 1
                            found_open = True
                        elif char == "}":
                            brace_count -= 1

                    if found_open and brace_count == 0:
                        block = "\n".join(lines[start:j + 1])
                        blocks.append((start + 1, j + 1, block))
                        i = j + 1
                        break
                else:
                    i += 1
            else:
                i += 1

    return blocks


def _guess_lang(file_path: str) -> str:
    """파일 확장자로 언어 추측"""
    fp = file_path.lower()
    if fp.endswith(".py"):
        return "py"
    if fp.endswith((".ts", ".tsx")):
        return "ts"
    if fp.endswith((".js", ".jsx")):
        return "js"
    if fp.endswith(".dart"):
        return "dart"
    return "unknown"


async def _search_files(
    repo_root: str,
    extensions: list[str],
    timeout_sec: float = 3.0,
) -> list[str]:
    """특정 확장자의 파일 목록 검색"""
    glob_patterns = []
    for ext in extensions:
        glob_patterns.extend(["--glob", f"**/*{ext}"])

    cmd = [
        "rg",
        "--files",
        "--color", "never",
        "--glob", "!.git/**",
        "--glob", "!node_modules/**",
        "--glob", "!dist/**",
        "--glob", "!build/**",
        "--glob", "!__pycache__/**",
        "--glob", "!*.min.js",
        "--glob", "!coverage/**",
        *glob_patterns,
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
        return []

    return stdout.decode("utf-8", errors="replace").splitlines()


async def find_similar_code(
    *,
    repo_root: str,
    query_code: str,
    exclude_files: Optional[list[str]] = None,
    min_similarity: float = 0.3,
    max_results: int = 5,
    ngram_size: int = 3,
    timeout_sec: float = 10.0,
) -> list[SimilarCodeMatch]:
    """
    쿼리 코드와 유사한 기존 코드를 찾음.

    Args:
        repo_root: 검색할 저장소 루트
        query_code: 비교할 새 코드
        exclude_files: 제외할 파일 목록 (예: diff의 파일)
        min_similarity: 최소 유사도 (0.0 ~ 1.0)
        max_results: 최대 결과 수
        ngram_size: n-gram 크기 (기본 3)
        timeout_sec: 전체 타임아웃

    Returns:
        SimilarCodeMatch 목록 (유사도 내림차순)
    """
    exclude_set = set(exclude_files or [])

    # 쿼리 코드 토큰화
    query_tokenized = _tokenize_code(query_code, ngram_size)

    if not query_tokenized.ngrams:
        return []

    # 검색할 파일 확장자 결정 (쿼리에서 추정하거나 모든 주요 언어)
    extensions = [".py", ".ts", ".tsx", ".js", ".jsx", ".dart"]

    # 파일 목록 가져오기
    files = await _search_files(repo_root, extensions, timeout_sec / 3)

    matches: list[SimilarCodeMatch] = []

    # 각 파일 처리
    for file_path in files:
        # 제외 파일 건너뛰기
        if file_path in exclude_set:
            continue

        full_path = Path(repo_root) / file_path
        if not full_path.exists():
            continue

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lang = _guess_lang(file_path)

        # 코드 블록 추출
        blocks = _extract_code_blocks(content, lang)

        for start_line, end_line, block in blocks:
            block_tokenized = _tokenize_code(block, ngram_size)

            if not block_tokenized.ngrams:
                continue

            # 유사도 계산
            similarity = _jaccard_similarity(query_tokenized.ngrams, block_tokenized.ngrams)

            if similarity >= min_similarity:
                matched_ngrams = len(query_tokenized.ngrams & block_tokenized.ngrams)

                # 스니펫 생성 (너무 길면 자름)
                snippet_lines = block.splitlines()
                if len(snippet_lines) > 15:
                    snippet = "\n".join(snippet_lines[:15]) + "\n..."
                else:
                    snippet = block

                matches.append(SimilarCodeMatch(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    snippet=snippet,
                    similarity=similarity,
                    matched_ngrams=matched_ngrams,
                    total_ngrams=len(query_tokenized.ngrams),
                    reason=f"Jaccard similarity {similarity:.2%} ({matched_ngrams}/{len(query_tokenized.ngrams)} n-grams)",
                ))

    # 유사도 내림차순 정렬
    matches.sort(key=lambda m: -m.similarity)

    return matches[:max_results]


async def find_similar_to_diff(
    *,
    repo_root: str,
    diff_text: str,
    min_similarity: float = 0.3,
    max_results: int = 5,
    ngram_size: int = 3,
) -> list[SimilarCodeMatch]:
    """
    diff에서 추가된 코드와 유사한 기존 코드를 찾음.

    Args:
        repo_root: 저장소 루트
        diff_text: git diff 텍스트
        min_similarity: 최소 유사도
        max_results: 최대 결과 수
        ngram_size: n-gram 크기

    Returns:
        SimilarCodeMatch 목록
    """
    # diff에서 추가된 라인만 추출
    added_lines: list[str] = []
    changed_files: list[str] = []
    current_file: Optional[str] = None

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            match = re.search(r"b/(.+)$", line)
            if match:
                current_file = match.group(1)
                changed_files.append(current_file)
        elif line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])  # '+' 제거

    if not added_lines:
        return []

    added_code = "\n".join(added_lines)

    return await find_similar_code(
        repo_root=repo_root,
        query_code=added_code,
        exclude_files=changed_files,
        min_similarity=min_similarity,
        max_results=max_results,
        ngram_size=ngram_size,
    )
