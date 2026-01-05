from __future__ import annotations

from backend.config.settings import settings
from backend.pipelines.base import ReviewPipeline
from backend.pipelines.evidence.refs_builder import build_refs_evidence
from backend.domain.tools.git_diff import get_git_diff, GitError


class MapreducePipeline(ReviewPipeline):
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
    
    async def build_review_payload(self, *, req, diff: str, diff_target: str) -> dict:
        payload = await super().build_review_payload(req=req, diff=diff, diff_target=diff_target)

        if getattr(req, "variant_id", "") in ("G1", "G1-evidence", "G1-baseline"):  # 네 규칙대로 조정
            refs = await build_refs_evidence(
                diff_text=diff,
                repo_root=".",           # repo root 경로(서버 실행 cwd 기준) / 설정으로 빼도 됨
                max_symbols=12,
                top_k_per_symbol=6,
            )
            payload["evidence_pack"] = {
                "refs": refs,
            }

        return payload