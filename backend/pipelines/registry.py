from __future__ import annotations
import importlib
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from backend.domain.prompts.registry import PromptPackRegistry
from backend.config.settings import settings


@dataclass(frozen=True)
class PipelineSpec:
    id: str
    pipeline: str           # "module.path:ClassName"
    params: Dict[str, Any]


# Default presets directory
_PRESETS_DIR = Path(__file__).parent / "presets"
_PACKS_DIR = Path(__file__).parent.parent / "domain" / "prompts" / "packs"


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


# =============================================================================
# Module-level convenience functions
# =============================================================================


def list_available_presets() -> List[Dict[str, Any]]:
    """
    List all available pipeline presets.

    Returns:
        List of preset configurations (id, pipeline, params)
    """
    presets = []
    for path in _PRESETS_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            presets.append({
                "id": data.get("id", path.stem),
                "pipeline": data.get("pipeline", ""),
                "params": data.get("params", {}),
            })
        except Exception:
            continue
    return sorted(presets, key=lambda p: p["id"].lower())


def load_preset(variant_id: str) -> Dict[str, Any]:
    """
    Load a preset configuration by variant ID.

    Args:
        variant_id: Variant ID (case-insensitive)

    Returns:
        Preset configuration dict

    Raises:
        FileNotFoundError: If preset not found
    """
    path = _PRESETS_DIR / f"{variant_id.lower()}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {variant_id}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "id": data.get("id", path.stem),
        "pipeline": data.get("pipeline", ""),
        "prompt_pack": data.get("prompt_pack"),  # optional: override prompt pack
        "params": data.get("params", {}),
    }


def get_pipeline(variant_id: str):
    """
    Get a configured pipeline instance for a variant.

    Args:
        variant_id: Variant ID (case-insensitive)

    Returns:
        Pipeline instance ready to run
    """
    # Load preset to check for prompt_pack override
    preset = load_preset(variant_id)
    prompt_pack_id = preset.get("prompt_pack") or variant_id

    # Load prompt pack
    prompt_registry = PromptPackRegistry(
        packs_dir=_PACKS_DIR,
        default_variant=settings.review_default_variant,
        allowed_variants=settings.review_allowed_variants,
    )
    pack = prompt_registry.get(prompt_pack_id)

    # Load pipeline spec and build
    registry = PipelineRegistry(str(_PRESETS_DIR))
    spec = registry.load_spec(variant_id)
    return registry.build_pipeline(spec, pack)
