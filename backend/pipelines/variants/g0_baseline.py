from __future__ import annotations

from backend.config.settings import settings
from backend.pipelines.base import ReviewPipeline
from backend.domain.tools.git_diff import get_git_diff, GitError


class BaselinePipeline(ReviewPipeline):
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
