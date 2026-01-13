"""
Utility functions for the evaluation webapp.

Provides helpers for:
- Loading datasets and variants
- Creating prompt snapshots
- Running evaluations
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from backend.evaluation.loader import load_dataset_by_name, list_available_datasets
from backend.evaluation.evaluator import Evaluator
from backend.evaluation.schemas import EvalRunResult, SampleScore
from backend.domain.schemas.review import ReviewRequest, ReviewResult


def get_available_datasets() -> list[dict[str, Any]]:
    """
    Get list of available datasets with metadata.

    Returns:
        List of dicts with name, sample_count, description
    """
    dataset_names = list_available_datasets()
    result = []
    for name in dataset_names:
        try:
            ds = load_dataset_by_name(name)
            result.append(
                {
                    "name": ds.name,
                    "sample_count": len(ds.samples),
                    "description": ds.description,
                    "version": ds.version,
                }
            )
        except Exception:
            # Skip invalid datasets
            continue
    return result


def get_available_variants() -> list[dict[str, Any]]:
    """
    Get list of available variants with their preset configs.

    Returns:
        List of dicts with id, description, params
    """
    from backend.pipelines.registry import list_available_presets

    presets = list_available_presets()
    result = []
    for preset in presets:
        result.append(
            {
                "id": preset["id"],
                "pipeline": preset.get("pipeline", ""),
                "params": preset.get("params", {}),
            }
        )
    return result


def get_variant_preset(variant_id: str) -> dict[str, Any]:
    """
    Get preset configuration for a variant.

    Args:
        variant_id: Variant ID (case-insensitive)

    Returns:
        Preset configuration dict
    """
    from backend.pipelines.registry import load_preset

    return load_preset(variant_id)


def get_prompt_pack(variant_id: str) -> dict[str, str]:
    """
    Get prompt pack content for a variant.

    Args:
        variant_id: Variant ID

    Returns:
        Dict with pack_id, review_system, review_user, etc.
    """
    from backend.domain.prompts.registry import PromptPackRegistry
    from backend.pipelines.registry import load_preset

    packs_dir = Path(__file__).parent.parent.parent / "domain" / "prompts" / "packs"

    # Check if preset overrides prompt_pack
    try:
        preset = load_preset(variant_id)
        prompt_pack_id = preset.get("prompt_pack") or variant_id
    except FileNotFoundError:
        prompt_pack_id = variant_id

    # Note: No allowed_variants restriction for evaluation webapp
    # We want to be able to load any prompt pack for testing/comparison
    registry = PromptPackRegistry(
        packs_dir=packs_dir,
        default_variant="g0-baseline",
        allowed_variants=None,  # Allow all packs
    )

    pack = registry.get(prompt_pack_id)
    return {
        "pack_id": pack.id,
        "review_system": pack.review_system,
        "review_user": pack.review_user,
        "repair_system": pack.repair_system,
        "repair_user": pack.repair_user,
    }


def create_prompt_snapshot(variant_id: str, include_content: bool = True):
    """
    Create a prompt snapshot for a variant.

    Args:
        variant_id: Variant ID
        include_content: Whether to include full prompt content

    Returns:
        PromptSnapshot instance
    """
    from backend.evaluation.webapp.storage.schemas import PromptSnapshot

    pack = get_prompt_pack(variant_id)
    return PromptSnapshot.from_prompt_pack(
        pack_id=pack["pack_id"],
        review_system=pack["review_system"],
        review_user=pack["review_user"],
        include_content=include_content,
    )


async def run_evaluation_async(
    dataset_name: str,
    variant_id: str,
    param_overrides: Optional[dict[str, Any]] = None,
    max_concurrency: int = 4,
    on_progress: Optional[Callable[[int, int, float], None]] = None,
) -> tuple[EvalRunResult, float]:
    """
    Run evaluation asynchronously.

    Args:
        dataset_name: Dataset to evaluate
        variant_id: Variant to use
        param_overrides: Parameter overrides (not yet implemented)
        max_concurrency: Max concurrent reviews
        on_progress: Progress callback (completed, total, current_f1)

    Returns:
        Tuple of (EvalRunResult, duration_seconds)
    """
    from backend.pipelines.registry import get_pipeline

    start_time = time.time()

    evaluator = Evaluator(dataset_name=dataset_name)
    total_samples = len(evaluator.dataset.samples)

    # Progress tracking
    completed = 0
    current_scores: list[SampleScore] = []

    def progress_callback(sample_id: str, score: SampleScore):
        nonlocal completed, current_scores
        completed += 1
        current_scores.append(score)

        # Calculate running F1
        if current_scores:
            total_tp = sum(s.true_positives for s in current_scores)
            total_fp = sum(s.false_positives for s in current_scores)
            total_fn = sum(s.false_negatives for s in current_scores)

            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
        else:
            f1 = 0.0

        if on_progress:
            on_progress(completed, total_samples, f1)

    # Create review function
    async def review_fn(diff: str, vid: str) -> ReviewResult:
        pipeline = get_pipeline(vid)
        req = ReviewRequest(diff=diff, variant_id=vid)
        return await pipeline.run(req)

    # Run evaluation
    result = await evaluator.run(
        review_fn=review_fn,
        variant_id=variant_id,
        max_concurrency=max_concurrency,
        on_sample_complete=progress_callback,
    )

    duration = time.time() - start_time
    return result, duration


def run_evaluation_sync(
    dataset_name: str,
    variant_id: str,
    param_overrides: Optional[dict[str, Any]] = None,
    max_concurrency: int = 4,
    on_progress: Optional[Callable[[int, int, float], None]] = None,
) -> tuple[EvalRunResult, float]:
    """
    Run evaluation synchronously (wrapper for async version).

    Args:
        dataset_name: Dataset to evaluate
        variant_id: Variant to use
        param_overrides: Parameter overrides
        max_concurrency: Max concurrent reviews
        on_progress: Progress callback

    Returns:
        Tuple of (EvalRunResult, duration_seconds)
    """
    return asyncio.run(
        run_evaluation_async(
            dataset_name=dataset_name,
            variant_id=variant_id,
            param_overrides=param_overrides,
            max_concurrency=max_concurrency,
            on_progress=on_progress,
        )
    )


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_percentage(value: float) -> str:
    """Format a 0-1 value as percentage."""
    return f"{value * 100:.1f}%"
