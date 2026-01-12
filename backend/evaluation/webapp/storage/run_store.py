"""
JSON file-based storage for evaluation runs.

Stores each run as a separate JSON file in the runs directory.
Provides listing, filtering, and basic CRUD operations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from backend.evaluation.webapp.storage.schemas import StoredRun, RunSummary


class RunStore:
    """
    File-based storage for evaluation runs.

    Each run is stored as {run_id}.json in the runs directory.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize run store.

        Args:
            data_dir: Base directory for data storage.
                     Defaults to backend/evaluation/data
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data"

        self.runs_dir = data_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _get_run_path(self, run_id: str) -> Path:
        """Get file path for a run."""
        return self.runs_dir / f"{run_id}.json"

    def save(self, run: StoredRun) -> str:
        """
        Save a run to storage.

        Args:
            run: The run to save

        Returns:
            The run_id
        """
        path = self._get_run_path(run.run_id)
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        return run.run_id

    def load(self, run_id: str) -> StoredRun:
        """
        Load a run from storage.

        Args:
            run_id: The run ID to load

        Returns:
            The loaded run

        Raises:
            FileNotFoundError: If run doesn't exist
        """
        path = self._get_run_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")

        data = json.loads(path.read_text(encoding="utf-8"))
        return StoredRun.model_validate(data)

    def exists(self, run_id: str) -> bool:
        """Check if a run exists."""
        return self._get_run_path(run_id).exists()

    def delete(self, run_id: str) -> bool:
        """
        Delete a run.

        Args:
            run_id: The run ID to delete

        Returns:
            True if deleted, False if not found
        """
        path = self._get_run_path(run_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_runs(
        self,
        variant_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RunSummary]:
        """
        List runs with optional filtering.

        Args:
            variant_id: Filter by variant (case-insensitive)
            dataset_name: Filter by dataset
            limit: Maximum runs to return
            offset: Number of runs to skip

        Returns:
            List of run summaries, sorted by created_at descending
        """
        summaries: list[RunSummary] = []

        # Load all runs (could be optimized with index file for large datasets)
        for path in self.runs_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                run = StoredRun.model_validate(data)
                summary = RunSummary.from_stored_run(run)

                # Apply filters
                if variant_id and summary.variant_id.lower() != variant_id.lower():
                    continue
                if dataset_name and summary.dataset_name != dataset_name:
                    continue

                summaries.append(summary)
            except Exception:
                # Skip invalid files
                continue

        # Sort by created_at descending (newest first)
        summaries.sort(key=lambda s: s.created_at, reverse=True)

        # Apply pagination
        return summaries[offset : offset + limit]

    def list_all_run_ids(self) -> list[str]:
        """Get all run IDs."""
        return [p.stem for p in self.runs_dir.glob("*.json")]

    def get_unique_variants(self) -> list[str]:
        """Get list of unique variant IDs from stored runs."""
        variants: set[str] = set()
        for path in self.runs_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                variant_id = data.get("config", {}).get("variant_id")
                if variant_id:
                    variants.add(variant_id)
            except Exception:
                continue
        return sorted(variants)

    def get_unique_datasets(self) -> list[str]:
        """Get list of unique dataset names from stored runs."""
        datasets: set[str] = set()
        for path in self.runs_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                dataset_name = data.get("config", {}).get("dataset_name")
                if dataset_name:
                    datasets.add(dataset_name)
            except Exception:
                continue
        return sorted(datasets)

    def update_tags(self, run_id: str, tags: list[str]) -> StoredRun:
        """
        Update tags for a run.

        Args:
            run_id: The run to update
            tags: New tags list

        Returns:
            Updated run
        """
        run = self.load(run_id)
        run.tags = tags
        self.save(run)
        return run

    def update_notes(self, run_id: str, notes: str) -> StoredRun:
        """
        Update notes for a run.

        Args:
            run_id: The run to update
            notes: New notes

        Returns:
            Updated run
        """
        run = self.load(run_id)
        run.notes = notes
        self.save(run)
        return run
