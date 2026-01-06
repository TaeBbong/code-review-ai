"""
Evaluation module for code review bot performance measurement.

This module provides:
- Evaluation schemas for datasets and results
- Dataset loader (YAML format)
- Scoring logic (precision/recall/F1)
- Evaluator for running experiments
- LangSmith integration helpers
"""

from backend.evaluation.schemas import (
    # Enums
    Difficulty,
    DataSource,
    # Input
    ContextFile,
    EvalInput,
    # Expected
    ExpectedIssue,
    ExpectedResult,
    # Sample
    SampleMetadata,
    EvalSample,
    EvalDataset,
    # Results
    IssueMatch,
    SampleScore,
    CategoryScore,
    EvalRunResult,
)

from backend.evaluation.loader import (
    load_dataset,
    load_dataset_by_name,
    list_available_datasets,
    filter_samples_by_category,
    filter_samples_by_difficulty,
)

from backend.evaluation.scorer import (
    score_sample,
    aggregate_by_category,
    calculate_overall_metrics,
    match_issue,
)

from backend.evaluation.evaluator import (
    Evaluator,
    create_langsmith_dataset_examples,
    create_langsmith_evaluator,
)

from backend.evaluation.langsmith_integration import (
    LangSmithEvaluator,
    create_review_evaluators,
)

__all__ = [
    # Enums
    "Difficulty",
    "DataSource",
    # Schemas
    "ContextFile",
    "EvalInput",
    "ExpectedIssue",
    "ExpectedResult",
    "SampleMetadata",
    "EvalSample",
    "EvalDataset",
    "IssueMatch",
    "SampleScore",
    "CategoryScore",
    "EvalRunResult",
    # Loader
    "load_dataset",
    "load_dataset_by_name",
    "list_available_datasets",
    "filter_samples_by_category",
    "filter_samples_by_difficulty",
    # Scorer
    "score_sample",
    "aggregate_by_category",
    "calculate_overall_metrics",
    "match_issue",
    # Evaluator
    "Evaluator",
    "create_langsmith_dataset_examples",
    "create_langsmith_evaluator",
    # LangSmith
    "LangSmithEvaluator",
    "create_review_evaluators",
]
