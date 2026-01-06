"""
LangSmith integration for code review bot evaluation.

This module provides:
- Dataset upload to LangSmith
- Experiment running with tracing
- Custom evaluators for precision/recall/F1
"""

from __future__ import annotations

import os
from typing import Optional, Callable, Any
from datetime import datetime, timezone

from langsmith import Client
from langsmith.evaluation import evaluate, EvaluationResults
from langsmith.schemas import Example, Run

from backend.domain.schemas.review import ReviewResult, ReviewRequest
from backend.evaluation.schemas import (
    EvalDataset,
    EvalSample,
    ExpectedIssue,
    ExpectedResult,
    SampleMetadata,
    EvalInput,
)
from backend.evaluation.loader import load_dataset_by_name
from backend.evaluation.scorer import score_sample


class LangSmithEvaluator:
    """
    LangSmith integration for running experiments.

    Usage:
        evaluator = LangSmithEvaluator()

        # Upload dataset (once)
        evaluator.upload_dataset("v1_initial")

        # Run experiment
        results = await evaluator.run_experiment(
            dataset_name="v1_initial",
            variant_id="g1-mapreduce",
        )
    """

    def __init__(self, client: Optional[Client] = None):
        """
        Initialize LangSmith evaluator.

        Args:
            client: LangSmith client (uses default if not provided)
        """
        self.client = client or Client()

    def upload_dataset(
        self,
        dataset_name: str,
        langsmith_dataset_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        """
        Upload local evaluation dataset to LangSmith.

        Args:
            dataset_name: Local dataset name (e.g., "v1_initial")
            langsmith_dataset_name: Name in LangSmith (defaults to local name)
            description: Dataset description

        Returns:
            LangSmith dataset ID
        """
        # Load local dataset
        dataset = load_dataset_by_name(dataset_name)

        ls_name = langsmith_dataset_name or f"code-review-eval-{dataset_name}"
        ls_description = description or dataset.description

        # Check if dataset already exists
        existing = list(self.client.list_datasets(dataset_name=ls_name))
        if existing:
            ls_dataset = existing[0]
            print(f"Dataset '{ls_name}' already exists (id={ls_dataset.id})")
        else:
            ls_dataset = self.client.create_dataset(
                dataset_name=ls_name,
                description=ls_description,
            )
            print(f"Created dataset '{ls_name}' (id={ls_dataset.id})")

        # Upload examples
        examples = self._convert_to_langsmith_examples(dataset)

        for example in examples:
            self.client.create_example(
                dataset_id=ls_dataset.id,
                inputs=example["inputs"],
                outputs=example["outputs"],
                metadata=example["metadata"],
            )

        print(f"Uploaded {len(examples)} examples")
        return str(ls_dataset.id)

    def _convert_to_langsmith_examples(self, dataset: EvalDataset) -> list[dict]:
        """Convert EvalDataset to LangSmith example format."""
        examples = []

        for sample in dataset.samples:
            example = {
                "inputs": {
                    "diff": sample.input.diff,
                    "sample_id": sample.id,
                },
                "outputs": {
                    "expected_issues": [
                        {
                            "category": issue.category.value,
                            "severity_min": issue.severity_min.value,
                            "title_keywords": issue.title_keywords,
                            "description_keywords": issue.description_keywords,
                            "issue_id": issue.issue_id,
                            "rationale": issue.rationale,
                        }
                        for issue in sample.expected.issues
                    ],
                    "min_issues": sample.expected.min_issues,
                    "max_issues": sample.expected.max_issues,
                    "expected_risk": (
                        sample.expected.expected_risk.value
                        if sample.expected.expected_risk
                        else None
                    ),
                    "should_have_blockers": sample.expected.should_have_blockers,
                },
                "metadata": {
                    "sample_id": sample.id,
                    "category": sample.metadata.primary_category.value,
                    "difficulty": sample.metadata.difficulty.value,
                    "tags": sample.metadata.tags,
                    "description": sample.metadata.description,
                },
            }
            examples.append(example)

        return examples

    async def run_experiment(
        self,
        dataset_name: str,
        variant_id: str,
        experiment_prefix: Optional[str] = None,
        max_concurrency: int = 4,
    ) -> EvaluationResults:
        """
        Run evaluation experiment on LangSmith.

        Args:
            dataset_name: LangSmith dataset name
            variant_id: Variant ID to evaluate
            experiment_prefix: Prefix for experiment name
            max_concurrency: Maximum concurrent evaluations

        Returns:
            LangSmith EvaluationResults
        """
        from backend.pipelines.registry import get_pipeline

        # Create target function
        async def target(inputs: dict) -> dict:
            diff = inputs["diff"]
            req = ReviewRequest(diff=diff, variant_id=variant_id)

            pipeline = get_pipeline(variant_id)
            result = await pipeline.run(req)

            return result.model_dump()

        # Create evaluators
        evaluators = [
            self._create_precision_evaluator(),
            self._create_recall_evaluator(),
            self._create_f1_evaluator(),
            self._create_issue_count_evaluator(),
        ]

        # Run experiment
        experiment_name = experiment_prefix or f"{variant_id}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

        results = evaluate(
            target,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_name,
            max_concurrency=max_concurrency,
        )

        return results

    def _reconstruct_sample(self, example: Example) -> EvalSample:
        """Reconstruct EvalSample from LangSmith Example."""
        from backend.domain.schemas.review import Category, Severity, RiskLevel
        from backend.evaluation.schemas import Difficulty, DataSource

        outputs = example.outputs or {}
        metadata = example.metadata or {}

        # Parse expected issues
        expected_issues = []
        for ei in outputs.get("expected_issues", []):
            expected_issues.append(ExpectedIssue(
                category=Category(ei["category"]),
                severity_min=Severity(ei.get("severity_min", "low")),
                title_keywords=ei.get("title_keywords", []),
                description_keywords=ei.get("description_keywords", []),
                issue_id=ei.get("issue_id", ""),
                rationale=ei.get("rationale", ""),
            ))

        expected_risk = None
        if outputs.get("expected_risk"):
            expected_risk = RiskLevel(outputs["expected_risk"])

        expected = ExpectedResult(
            issues=expected_issues,
            min_issues=outputs.get("min_issues", 0),
            max_issues=outputs.get("max_issues"),
            expected_risk=expected_risk,
            should_have_blockers=outputs.get("should_have_blockers"),
        )

        sample_metadata = SampleMetadata(
            source=DataSource.synthetic,
            difficulty=Difficulty(metadata.get("difficulty", "medium")),
            primary_category=Category(metadata.get("category", "correctness")),
            tags=metadata.get("tags", []),
            description=metadata.get("description", ""),
        )

        return EvalSample(
            id=metadata.get("sample_id", example.id),
            input=EvalInput(diff=example.inputs.get("diff", "")),
            expected=expected,
            metadata=sample_metadata,
        )

    def _create_precision_evaluator(self) -> Callable:
        """Create precision evaluator."""
        def precision_evaluator(run: Run, example: Example) -> dict:
            sample = self._reconstruct_sample(example)
            prediction = ReviewResult.model_validate(run.outputs)
            score = score_sample(sample, prediction)

            return {
                "key": "precision",
                "score": score.precision,
            }

        return precision_evaluator

    def _create_recall_evaluator(self) -> Callable:
        """Create recall evaluator."""
        def recall_evaluator(run: Run, example: Example) -> dict:
            sample = self._reconstruct_sample(example)
            prediction = ReviewResult.model_validate(run.outputs)
            score = score_sample(sample, prediction)

            return {
                "key": "recall",
                "score": score.recall,
            }

        return recall_evaluator

    def _create_f1_evaluator(self) -> Callable:
        """Create F1 score evaluator."""
        def f1_evaluator(run: Run, example: Example) -> dict:
            sample = self._reconstruct_sample(example)
            prediction = ReviewResult.model_validate(run.outputs)
            score = score_sample(sample, prediction)

            return {
                "key": "f1_score",
                "score": score.f1_score,
            }

        return f1_evaluator

    def _create_issue_count_evaluator(self) -> Callable:
        """Create issue count evaluator (checks min/max bounds)."""
        def issue_count_evaluator(run: Run, example: Example) -> dict:
            outputs = example.outputs or {}
            prediction = ReviewResult.model_validate(run.outputs)

            issue_count = len(prediction.issues)
            min_issues = outputs.get("min_issues", 0)
            max_issues = outputs.get("max_issues")

            # Check if within bounds
            within_bounds = issue_count >= min_issues
            if max_issues is not None:
                within_bounds = within_bounds and issue_count <= max_issues

            return {
                "key": "issue_count_valid",
                "score": 1.0 if within_bounds else 0.0,
                "comment": f"Found {issue_count} issues (expected: {min_issues}-{max_issues or 'inf'})",
            }

        return issue_count_evaluator


# =============================================================================
# Standalone Evaluator Functions (for use with langsmith.evaluate directly)
# =============================================================================


def create_review_evaluators() -> list[Callable]:
    """
    Create list of evaluator functions for use with langsmith.evaluate().

    Usage:
        from langsmith.evaluation import evaluate
        from backend.evaluation.langsmith_integration import create_review_evaluators

        results = evaluate(
            target=my_review_function,
            data="code-review-eval-v1_initial",
            evaluators=create_review_evaluators(),
        )
    """
    def _reconstruct_sample_from_example(example: Example) -> EvalSample:
        """Helper to reconstruct sample from example."""
        from backend.domain.schemas.review import Category, Severity, RiskLevel
        from backend.evaluation.schemas import Difficulty, DataSource

        outputs = example.outputs or {}
        metadata = example.metadata or {}

        expected_issues = []
        for ei in outputs.get("expected_issues", []):
            expected_issues.append(ExpectedIssue(
                category=Category(ei["category"]),
                severity_min=Severity(ei.get("severity_min", "low")),
                title_keywords=ei.get("title_keywords", []),
                description_keywords=ei.get("description_keywords", []),
                issue_id=ei.get("issue_id", ""),
            ))

        expected_risk = None
        if outputs.get("expected_risk"):
            expected_risk = RiskLevel(outputs["expected_risk"])

        return EvalSample(
            id=metadata.get("sample_id", str(example.id)),
            input=EvalInput(diff=example.inputs.get("diff", "")),
            expected=ExpectedResult(
                issues=expected_issues,
                min_issues=outputs.get("min_issues", 0),
                max_issues=outputs.get("max_issues"),
                expected_risk=expected_risk,
                should_have_blockers=outputs.get("should_have_blockers"),
            ),
            metadata=SampleMetadata(
                primary_category=Category(metadata.get("category", "correctness")),
                difficulty=Difficulty(metadata.get("difficulty", "medium")),
            ),
        )

    def precision(run: Run, example: Example) -> dict:
        sample = _reconstruct_sample_from_example(example)
        prediction = ReviewResult.model_validate(run.outputs)
        score = score_sample(sample, prediction)
        return {"key": "precision", "score": score.precision}

    def recall(run: Run, example: Example) -> dict:
        sample = _reconstruct_sample_from_example(example)
        prediction = ReviewResult.model_validate(run.outputs)
        score = score_sample(sample, prediction)
        return {"key": "recall", "score": score.recall}

    def f1(run: Run, example: Example) -> dict:
        sample = _reconstruct_sample_from_example(example)
        prediction = ReviewResult.model_validate(run.outputs)
        score = score_sample(sample, prediction)
        return {"key": "f1_score", "score": score.f1_score}

    def tp_fp_fn(run: Run, example: Example) -> dict:
        sample = _reconstruct_sample_from_example(example)
        prediction = ReviewResult.model_validate(run.outputs)
        score = score_sample(sample, prediction)
        return {
            "key": "detection_counts",
            "score": score.true_positives,  # Use TP as primary score
            "comment": f"TP={score.true_positives}, FP={score.false_positives}, FN={score.false_negatives}",
        }

    return [precision, recall, f1, tp_fp_fn]
