from __future__ import annotations

import json
from typing import Any, Dict, List

from backend.config.settings import settings
from backend.pipelines.base import ReviewPipeline
from backend.pipelines.evidence.refs_builder import build_refs_evidence
from backend.domain.schemas.diff import DiffChunk
from backend.domain.schemas.review import ReviewRequest
from backend.domain.tools.git_diff import get_git_diff, GitError, chunk_diff_by_file


class MapReducePipeline(ReviewPipeline):
    """
    G1-mapreduce: 파일별 분할 리뷰 + ripgrep 심볼 레퍼런스 수집 파이프라인.

    핵심 기능:
    - diff를 파일별로 분할하여 병렬 리뷰 (map-reduce)
    - diff에서 변경된 심볼(함수, 클래스 등) 추출
    - ripgrep으로 해당 심볼의 레퍼런스를 검색
    - evidence_pack.refs에 검색 결과를 포함하여 LLM에 전달
    """

    async def resolve_diff(self, req: ReviewRequest) -> tuple[str, str]:
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

    async def split_chunks(self, diff: str) -> List[DiffChunk]:
        """
        diff를 파일별로 분할.
        파일 수가 min_files_for_split 미만이면 분할하지 않음.
        """
        if not diff.strip():
            return [DiffChunk(file_path="", content=diff)]

        min_files = int(self.params.get("min_files_for_split", 2))
        max_files = int(self.params.get("max_files_for_split", 20))

        file_chunks = chunk_diff_by_file(diff)

        # 파일 수가 적으면 분할 안 함 (단일 LLM 호출이 더 효율적)
        if len(file_chunks) < min_files:
            return [DiffChunk(file_path="", content=diff)]

        # 파일 수가 너무 많으면 상위 N개만 (토큰 제한)
        if len(file_chunks) > max_files:
            file_chunks = file_chunks[:max_files]

        chunks = [
            DiffChunk(
                file_path=fc["file_b"],
                content=fc["diff"],
                metadata={"file_a": fc["file_a"]},
            )
            for fc in file_chunks
        ]

        return chunks

    async def build_evidence(
        self,
        *,
        req: ReviewRequest,
        diff: str,
    ) -> Dict[str, Any]:
        """
        ripgrep으로 심볼 레퍼런스를 수집.
        수집된 evidence는 build_review_payload()에서 LLM에 전달됨.
        """
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

        return {"refs": refs}

    async def build_review_payload(
        self,
        *,
        req: ReviewRequest,
        diff: str,
        diff_target: str,
        chunk: DiffChunk | None = None,
    ) -> Dict[str, Any]:
        """
        LLM에 전달할 payload를 구성.
        evidence_pack에 수집된 refs를 포함.
        """
        payload = await super().build_review_payload(
            req=req,
            diff=diff,
            diff_target=diff_target,
            chunk=chunk,
        )

        # self._evidence_pack은 build_evidence()에서 채워짐
        refs = self._evidence_pack.get("refs", [])

        payload["evidence_pack"] = {
            "refs": refs,
        }

        # 프롬프트에서 사용할 수 있도록 문자열로도 제공
        if refs:
            payload["evidence_refs_json"] = json.dumps(refs, ensure_ascii=False, indent=2)
        else:
            payload["evidence_refs_json"] = "[]"

        return payload
