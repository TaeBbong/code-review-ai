"""
Storage schemas for evaluation run persistence.

These schemas extend the base evaluation schemas to include:
- Configuration snapshots (params, prompts)
- Execution metadata (duration, tags)
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict

from backend.evaluation.schemas import EvalRunResult


class PromptSnapshot(BaseModel):
    """Snapshot of prompts used during evaluation."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str = Field(..., description="Prompt pack ID")
    review_system_hash: str = Field(..., description="SHA256 hash of review system prompt")
    review_user_hash: str = Field(..., description="SHA256 hash of review user prompt")

    # Full content (optional, for detailed comparison)
    review_system_content: Optional[str] = Field(
        default=None, description="Full review system prompt content"
    )
    review_user_content: Optional[str] = Field(
        default=None, description="Full review user prompt content"
    )

    @classmethod
    def from_prompt_pack(
        cls,
        pack_id: str,
        review_system: str,
        review_user: str,
        include_content: bool = True,
    ) -> "PromptSnapshot":
        """Create snapshot from prompt pack content."""
        return cls(
            pack_id=pack_id,
            review_system_hash=hashlib.sha256(review_system.encode()).hexdigest()[:16],
            review_user_hash=hashlib.sha256(review_user.encode()).hexdigest()[:16],
            review_system_content=review_system if include_content else None,
            review_user_content=review_user if include_content else None,
        )


class RunConfig(BaseModel):
    """Configuration snapshot for an evaluation run."""

    model_config = ConfigDict(extra="forbid")

    variant_id: str = Field(..., description="Variant ID used")
    dataset_name: str = Field(..., description="Dataset name")

    # Full preset parameters
    preset_params: dict[str, Any] = Field(
        default_factory=dict, description="All preset parameters"
    )

    # User overrides
    overrides: dict[str, Any] = Field(
        default_factory=dict, description="User-specified parameter overrides"
    )

    # Prompt snapshot
    prompt_snapshot: PromptSnapshot = Field(..., description="Prompts used")

    # Evaluation settings
    max_concurrency: int = Field(default=4, description="Max concurrent evaluations")

    def get_effective_params(self) -> dict[str, Any]:
        """Get effective parameters (preset + overrides)."""
        effective = dict(self.preset_params)
        effective.update(self.overrides)
        return effective


class StoredRun(BaseModel):
    """Complete stored evaluation run with metadata."""

    model_config = ConfigDict(extra="forbid")

    # Basic info
    run_id: str = Field(..., description="Unique run identifier")
    created_at: str = Field(..., description="ISO timestamp of run start")
    duration_seconds: float = Field(..., description="Total execution time")

    # Configuration snapshot
    config: RunConfig = Field(..., description="Configuration used for this run")

    # Results (from EvalRunResult)
    result: EvalRunResult = Field(..., description="Evaluation results")

    # User metadata
    tags: list[str] = Field(default_factory=list, description="User-defined tags")
    notes: str = Field(default="", description="User notes")


class RunSummary(BaseModel):
    """Lightweight summary for run listing."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: str
    duration_seconds: float

    variant_id: str
    dataset_name: str

    # Key metrics
    total_samples: int
    overall_precision: float
    overall_recall: float
    overall_f1: float

    tags: list[str] = Field(default_factory=list)

    @classmethod
    def from_stored_run(cls, run: StoredRun) -> "RunSummary":
        """Create summary from full stored run."""
        return cls(
            run_id=run.run_id,
            created_at=run.created_at,
            duration_seconds=run.duration_seconds,
            variant_id=run.config.variant_id,
            dataset_name=run.config.dataset_name,
            total_samples=run.result.total_samples,
            overall_precision=run.result.overall_precision,
            overall_recall=run.result.overall_recall,
            overall_f1=run.result.overall_f1,
            tags=run.tags,
        )
