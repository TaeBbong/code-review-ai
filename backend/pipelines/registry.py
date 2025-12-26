from __future__ import annotations
import importlib
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

@dataclass(frozen=True)
class PipelineSpec:
    id: str
    pipeline: str           # "module.path:ClassName"
    params: Dict[str, Any]


class PipelineRegistry:
    def __init__(self, presets_dir: str):
        self.presets_dir = Path(presets_dir)

    def load_spec(self, variant_id: str) -> PipelineSpec:
        path = self.presets_dir / f"{variant_id.lower()}.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return PipelineSpec(id=data["id"], pipeline=data["pipeline"], params=data.get("params", {}))

    def build_pipeline(self, spec: PipelineSpec, pack):
        mod_path, cls_name = spec.pipeline.split(":")
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        return cls(pack=pack, params=spec.params)
