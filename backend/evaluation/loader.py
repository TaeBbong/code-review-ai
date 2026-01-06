"""
Dataset loader for evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from backend.evaluation.schemas import (
    EvalDataset,
    EvalSample,
    EvalInput,
    ExpectedResult,
    ExpectedIssue,
    SampleMetadata,
    ContextFile,
    Difficulty,
    DataSource,
)
from backend.domain.schemas.review import Category, Severity, RiskLevel


def _parse_category(value: str) -> Category:
    """Parse category string to enum."""
    # Handle hyphenated values like 'api-compat'
    return Category(value)


def _parse_severity(value: str) -> Severity:
    """Parse severity string to enum."""
    return Severity(value)


def _parse_risk_level(value: str) -> RiskLevel:
    """Parse risk level string to enum."""
    return RiskLevel(value)


def _parse_difficulty(value: str) -> Difficulty:
    """Parse difficulty string to enum."""
    return Difficulty(value)


def _parse_data_source(value: str) -> DataSource:
    """Parse data source string to enum."""
    return DataSource(value)


def _parse_expected_issue(data: dict) -> ExpectedIssue:
    """Parse expected issue from dict."""
    return ExpectedIssue(
        category=_parse_category(data["category"]),
        severity_min=_parse_severity(data.get("severity_min", "low")),
        file_pattern=data.get("file_pattern"),
        line_start=data.get("line_start"),
        line_tolerance=data.get("line_tolerance", 3),
        title_keywords=data.get("title_keywords", []),
        description_keywords=data.get("description_keywords", []),
        issue_id=data.get("issue_id", ""),
        rationale=data.get("rationale", ""),
    )


def _parse_expected_result(data: dict) -> ExpectedResult:
    """Parse expected result from dict."""
    issues = [_parse_expected_issue(i) for i in data.get("issues", [])]

    forbidden_categories = [
        _parse_category(c) for c in data.get("forbidden_categories", [])
    ]

    expected_risk = None
    if data.get("expected_risk"):
        expected_risk = _parse_risk_level(data["expected_risk"])

    return ExpectedResult(
        issues=issues,
        min_issues=data.get("min_issues", 0),
        max_issues=data.get("max_issues"),
        expected_risk=expected_risk,
        should_have_blockers=data.get("should_have_blockers"),
        forbidden_categories=forbidden_categories,
    )


def _parse_context_file(data: dict) -> ContextFile:
    """Parse context file from dict."""
    return ContextFile(
        path=data["path"],
        content=data["content"],
    )


def _parse_eval_input(data: dict) -> EvalInput:
    """Parse eval input from dict."""
    context_files = [
        _parse_context_file(cf) for cf in data.get("context_files", [])
    ]
    return EvalInput(
        diff=data["diff"],
        context_files=context_files,
    )


def _parse_sample_metadata(data: dict) -> SampleMetadata:
    """Parse sample metadata from dict."""
    return SampleMetadata(
        source=_parse_data_source(data.get("source", "synthetic")),
        difficulty=_parse_difficulty(data.get("difficulty", "medium")),
        primary_category=_parse_category(data["primary_category"]),
        tags=data.get("tags", []),
        description=data.get("description", ""),
        created_at=data.get("created_at", ""),
        author=data.get("author", ""),
    )


def _parse_eval_sample(data: dict) -> EvalSample:
    """Parse evaluation sample from dict."""
    return EvalSample(
        id=data["id"],
        input=_parse_eval_input(data["input"]),
        expected=_parse_expected_result(data["expected"]),
        metadata=_parse_sample_metadata(data["metadata"]),
    )


def load_dataset(path: Path | str) -> EvalDataset:
    """
    Load evaluation dataset from YAML file.

    Args:
        path: Path to the YAML dataset file

    Returns:
        Parsed EvalDataset
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    samples = [_parse_eval_sample(s) for s in data.get("samples", [])]

    return EvalDataset(
        name=data.get("name", path.stem),
        version=data.get("version", "1.0.0"),
        description=data.get("description", ""),
        samples=samples,
    )


def load_dataset_by_name(name: str) -> EvalDataset:
    """
    Load evaluation dataset by name from the datasets directory.

    Args:
        name: Dataset name (without extension)

    Returns:
        Parsed EvalDataset
    """
    datasets_dir = Path(__file__).parent / "datasets"
    path = datasets_dir / f"{name}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return load_dataset(path)


def list_available_datasets() -> list[str]:
    """
    List all available dataset names.

    Returns:
        List of dataset names
    """
    datasets_dir = Path(__file__).parent / "datasets"
    return [p.stem for p in datasets_dir.glob("*.yaml")]


def filter_samples_by_category(
    dataset: EvalDataset,
    category: Category,
) -> list[EvalSample]:
    """
    Filter samples by primary category.

    Args:
        dataset: The dataset to filter
        category: Category to filter by

    Returns:
        Filtered list of samples
    """
    return [
        s for s in dataset.samples
        if s.metadata.primary_category == category
    ]


def filter_samples_by_difficulty(
    dataset: EvalDataset,
    difficulty: Difficulty,
) -> list[EvalSample]:
    """
    Filter samples by difficulty.

    Args:
        dataset: The dataset to filter
        difficulty: Difficulty to filter by

    Returns:
        Filtered list of samples
    """
    return [
        s for s in dataset.samples
        if s.metadata.difficulty == difficulty
    ]
