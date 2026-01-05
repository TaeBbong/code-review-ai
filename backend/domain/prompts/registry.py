from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import json


@dataclass(frozen=True)
class PromptPack:
    id: str
    description: str
    review_system: str
    review_user: str
    repair_system: str
    repair_user: str
    params: dict[str, Any]


class PromptPackNotFound(RuntimeError):
    pass


class PromptPackRegistry:
    """
    packs_dir/
      baseline/
        manifest.yaml(or json)
        review.system.txt
        review.user.txt
        repair.system.txt
        repair.user.txt
    """

    def __init__(
        self,
        packs_dir: Path,
        default_variant: str = "baseline",
        allowed_variants: tuple[str, ...] | None = None,
    ):
        self.packs_dir = packs_dir
        self.default_variant = default_variant
        self.allowed_variants = allowed_variants

    def resolve_variant(self, variant_id: str | None) -> str:
        vid = (variant_id or "").strip() or self.default_variant
        if self.allowed_variants:
            # 대소문자 구분 없이 allowed_variants 체크
            allowed_lower = {v.lower() for v in self.allowed_variants}
            if vid.lower() not in allowed_lower:
                # 운영 정책: 허용되지 않은 variant면 default로 폴백
                return self.default_variant
        return vid

    def get(self, variant_id: str | None) -> PromptPack:
        vid = self.resolve_variant(variant_id)
        return self._load_pack_cached(str(self.packs_dir), vid)

    @staticmethod
    @lru_cache(maxsize=64)
    def _load_pack_cached(packs_dir_str: str, vid: str) -> PromptPack:
        packs_dir = Path(packs_dir_str)
        pack_dir = packs_dir / vid

        # 대소문자 구분 없이 폴더 찾기 (Linux 호환)
        if not pack_dir.exists():
            vid_lower = vid.lower()
            for candidate in packs_dir.iterdir():
                if candidate.is_dir() and candidate.name.lower() == vid_lower:
                    pack_dir = candidate
                    break

        if not pack_dir.exists():
            raise PromptPackNotFound(f"Prompt pack not found: {pack_dir}")

        manifest = _load_manifest(pack_dir)
        templates = manifest.get("templates", {})

        def read_template(key: str, fallback_filename: str) -> str:
            filename = templates.get(key, fallback_filename)
            return (pack_dir / filename).read_text(encoding="utf-8")

        return PromptPack(
            id=str(manifest.get("id", vid)),
            description=str(manifest.get("description", "")),
            review_system=read_template("review_system", "review.system.txt"),
            review_user=read_template("review_user", "review.user.txt"),
            repair_system=read_template("repair_system", "repair.system.txt"),
            repair_user=read_template("repair_user", "repair.user.txt"),
            params=dict(manifest.get("params", {})),
        )


def _load_manifest(pack_dir: Path) -> dict[str, Any]:
    """
    - manifest.yaml이 있으면 YAML 우선
    - 없으면 manifest.json 사용
    (PyYAML이 없을 수도 있으니 안전하게 처리)
    """
    yaml_path = pack_dir / "manifest.yaml"
    yml_path = pack_dir / "manifest.yml"
    json_path = pack_dir / "manifest.json"

    if yaml_path.exists() or yml_path.exists():
        path = yaml_path if yaml_path.exists() else yml_path
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"Found {path.name} but PyYAML is not installed. "
                f"Either install pyyaml or provide manifest.json."
            ) from e
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError(f"Invalid manifest format: {path}")
        return data

    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError(f"Invalid manifest format: {json_path}")
        return data

    # manifest 없이도 동작하게(최소)
    return {"id": pack_dir.name, "description": "", "templates": {}, "params": {}}
