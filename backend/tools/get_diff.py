from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class GitError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiffStats:
    files: int
    insertions: int
    deletions: int
    per_file: List[Dict[str, Any]]  # [{"path": "...", "insertions": 1, "deletions": 2}]


def _run_git(args: List[str], cwd: Optional[str] = None, timeout: int = 20) -> Tuple[int, str, str]:
    """
    Returns (returncode, stdout, stderr)
    """
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        shell=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _find_repo_root(start: Optional[str] = None) -> str:
    cwd = start or os.getcwd()
    code, out, err = _run_git(["rev-parse", "--show-toplevel"], cwd=cwd)
    if code != 0:
        raise GitError(f"Not a git repository (cwd={cwd}). git error: {err.strip()}")
    return out.strip()


def get_git_diff(
    *,
    diff_target: str = "staged",
    repo_path: Optional[str] = None,
    context_lines: int = 3,
    max_chars: int = 1_500_000,
) -> str:
    """
    diff_target:
      - "staged" (default): git diff --staged
      - "worktree": git diff
      - "<commit-ish>..<commit-ish>" : git diff A..B
      - "<commit-ish>...<commit-ish>": git diff A...B (merge-base compare)
    """
    root = repo_path or _find_repo_root(None)

    base_args = ["diff", f"--unified={context_lines}"]
    if diff_target == "staged":
        args = [*base_args, "--staged"]
    elif diff_target == "worktree":
        args = base_args
    elif ".." in diff_target or "..." in diff_target:
        # commit range / compare
        args = [*base_args, diff_target]
    else:
        # fallback: treat as staged for safety
        args = [*base_args, "--staged"]

    code, out, err = _run_git(args, cwd=root)
    if code != 0:
        raise GitError(f"git diff failed: {err.strip()}")

    out = out if out is not None else ""
    if len(out) > max_chars:
        # hard cut to prevent blowing context; later Day6에서 더 정교하게 컨텍스트 예산 처리
        out = out[:max_chars] + "\n\n# [TRUNCATED]\n"
    return out


def get_git_numstat(
    *,
    diff_target: str = "staged",
    repo_path: Optional[str] = None,
) -> str:
    root = repo_path or _find_repo_root(None)

    base_args = ["diff", "--numstat"]
    if diff_target == "staged":
        args = [*base_args, "--staged"]
    elif diff_target == "worktree":
        args = base_args
    elif ".." in diff_target or "..." in diff_target:
        args = [*base_args, diff_target]
    else:
        args = [*base_args, "--staged"]

    code, out, err = _run_git(args, cwd=root)
    if code != 0:
        raise GitError(f"git diff --numstat failed: {err.strip()}")
    return out or ""


def parse_diff_stats(numstat_text: str) -> DiffStats:
    """
    git diff --numstat output format:
      <ins>\t<del>\t<path>
    where ins/del can be "-" for binary.
    """
    per_file: List[Dict[str, Any]] = []
    total_ins = 0
    total_del = 0

    for line in (numstat_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue

        ins_raw, del_raw, path = parts[0], parts[1], parts[2]
        ins = int(ins_raw) if ins_raw.isdigit() else 0
        dele = int(del_raw) if del_raw.isdigit() else 0

        total_ins += ins
        total_del += dele
        per_file.append({"path": path, "insertions": ins, "deletions": dele})

    return DiffStats(files=len(per_file), insertions=total_ins, deletions=total_del, per_file=per_file)


_DIFF_START_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$", re.MULTILINE)


def chunk_diff_by_file(diff_text: str) -> List[Dict[str, str]]:
    """
    Splits unified diff into per-file chunks based on 'diff --git a/... b/...'
    Returns: [{"file_a": "...", "file_b": "...", "diff": "<chunk>"}...]
    """
    text = diff_text or ""
    matches = list(_DIFF_START_RE.finditer(text))
    if not matches:
        return []

    chunks: List[Dict[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        file_a, file_b = m.group(1), m.group(2)
        chunk = text[start:end].rstrip() + "\n"
        chunks.append({"file_a": file_a, "file_b": file_b, "diff": chunk})
    return chunks
