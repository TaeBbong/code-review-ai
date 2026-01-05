from __future__ import annotations

import json
from typing import Any, Dict

from backend.config.settings import settings
from backend.pipelines.base import ReviewPipeline
from backend.pipelines.evidence.refs_builder import build_refs_evidence
from backend.domain.tools.git_diff import get_git_diff, GitError


class MapReducePipeline(ReviewPipeline):
    """
    G1-mapreduce: ripgrep을 활용한 심볼 레퍼런스 수집 파이프라인.

    핵심 기능:
    - diff에서 변경된 심볼(함수, 클래스 등) 추출
    - ripgrep으로 해당 심볼의 레퍼런스를 검색
    - evidence_pack.refs에 검색 결과를 포함하여 LLM에 전달
    """

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

    async def build_review_payload(self, *, req, diff: str, diff_target: str) -> Dict[str, Any]:
        """
        LLM에 전달할 payload를 구성.
        evidence_pack.refs에 ripgrep 검색 결과를 포함.
        """
        payload = await super().build_review_payload(req=req, diff=diff, diff_target=diff_target)

        # repo_root 결정: settings > params > current directory
        repo_root = (
            str(settings.review_repo_path) if settings.review_repo_path
            else self.params.get("repo_root", ".")
        )

        max_symbols = int(self.params.get("max_symbols", 12))
        top_k_per_symbol = int(self.params.get("top_k_per_symbol", 6))

        refs = await build_refs_evidence(
            diff_text=diff,
            repo_root=repo_root,
            max_symbols=max_symbols,
            top_k_per_symbol=top_k_per_symbol,
        )

        payload["evidence_pack"] = {
            "refs": refs,
        }

        # 프롬프트에서 사용할 수 있도록 문자열로도 제공
        if refs:
            payload["evidence_refs_json"] = json.dumps(refs, ensure_ascii=False, indent=2)
        else:
            payload["evidence_refs_json"] = "[]"

        return payload