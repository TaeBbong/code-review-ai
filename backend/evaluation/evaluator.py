"""
Main evaluator that orchestrates the evaluation process.

Runs review pipelines on evaluation samples and collects metrics.
Designed for LangSmith integration.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional, Callable, Any

from backend.domain.schemas.review import ReviewResult, ReviewRequest
from backend.evaluation.schemas import (
    EvalDataset,
    EvalSample,
    EvalRunResult,
    SampleScore,
)
from backend.evaluation.scorer import (
    score_sample,
    aggregate_by_category,
    calculate_overall_metrics,
)
from backend.evaluation.loader import load_dataset, load_dataset_by_name


# Type alias for review function
ReviewFn = Callable[[str, str], ReviewResult]  # (diff, variant_id) -> ReviewResult
AsyncReviewFn = Callable[[str, str], Any]  # async version


class Evaluator:
    """
    Evaluator for code review bot performance measurement.

    Designed to work with LangSmith for experiment tracking.

    Usage:
        evaluator = Evaluator(dataset_name="v1_initial")

        # Define review function (adapter to your pipeline)
        async def review_fn(diff: str, variant_id: str) -> ReviewResult:
            return await pipeline.run(diff, variant_id)

        # Run evaluation
        result = await evaluator.run(
            review_fn=review_fn,
            variant_id="g1-mapreduce",
        )
    """

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        dataset: Optional[EvalDataset] = None,
    ):
        """
        Initialize evaluator.

        Args:
            dataset_name: Name of dataset to load from datasets directory
            dataset: Pre-loaded dataset (alternative to dataset_name)
        """
        if dataset is not None:
            self.dataset = dataset
        elif dataset_name is not None:
            self.dataset = load_dataset_by_name(dataset_name)
        else:
            raise ValueError("Either dataset_name or dataset must be provided")

    async def run(
        self,
        review_fn: AsyncReviewFn,
        variant_id: str,
        run_id: Optional[str] = None,
        sample_ids: Optional[list[str]] = None,
        max_concurrency: int = 1,
        on_sample_complete: Optional[Callable[[str, SampleScore], None]] = None,
    ) -> EvalRunResult:
        """
        Run evaluation on all samples.

        Args:
            review_fn: Async function that takes (diff, variant_id) and returns ReviewResult
            variant_id: Variant ID to evaluate
            run_id: Optional run ID (auto-generated if not provided)
            sample_ids: Optional list of specific sample IDs to evaluate
            max_concurrency: Maximum concurrent reviews
            on_sample_complete: Optional callback for progress tracking

        Returns:
            EvalRunResult with all metrics
        """
        run_id = run_id or str(uuid.uuid4())[:8]

        # Filter samples if specific IDs provided
        samples = self.dataset.samples
        if sample_ids:
            samples = [s for s in samples if s.id in sample_ids]

        # Run evaluations with concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)
        predictions: dict[str, ReviewResult] = {}
        sample_scores: list[SampleScore] = []

        async def evaluate_sample(sample: EvalSample) -> tuple[str, ReviewResult, SampleScore]:
            async with semaphore:
                # Call review function
                result = await review_fn(sample.input.diff, variant_id)

                # Score the result
                score = score_sample(sample, result)

                if on_sample_complete:
                    on_sample_complete(sample.id, score)

                return sample.id, result, score

        # Run all samples
        tasks = [evaluate_sample(s) for s in samples]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                # Log error but continue
                print(f"Error evaluating sample: {result}")
                continue

            sample_id, prediction, score = result
            predictions[sample_id] = prediction
            sample_scores.append(score)

        # Aggregate metrics
        category_scores = aggregate_by_category(samples, sample_scores)
        overall = calculate_overall_metrics(sample_scores)

        return EvalRunResult(
            run_id=run_id,
            dataset_name=self.dataset.name,
            variant_id=variant_id,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            sample_scores=sample_scores,
            category_scores=category_scores,
            total_samples=len(sample_scores),
            total_tp=overall["total_tp"],
            total_fp=overall["total_fp"],
            total_fn=overall["total_fn"],
            overall_precision=overall["precision"],
            overall_recall=overall["recall"],
            overall_f1=overall["f1"],
            predictions=predictions,
        )

    def run_sync(
        self,
        review_fn: ReviewFn,
        variant_id: str,
        run_id: Optional[str] = None,
        sample_ids: Optional[list[str]] = None,
    ) -> EvalRunResult:
        """
        Synchronous version of run() for simpler use cases.

        Args:
            review_fn: Sync function that takes (diff, variant_id) and returns ReviewResult
            variant_id: Variant ID to evaluate
            run_id: Optional run ID
            sample_ids: Optional list of specific sample IDs

        Returns:
            EvalRunResult with all metrics
        """
        run_id = run_id or str(uuid.uuid4())[:8]

        samples = self.dataset.samples
        if sample_ids:
            samples = [s for s in samples if s.id in sample_ids]

        predictions: dict[str, ReviewResult] = {}
        sample_scores: list[SampleScore] = []

        for sample in samples:
            try:
                result = review_fn(sample.input.diff, variant_id)
                score = score_sample(sample, result)
                predictions[sample.id] = result
                sample_scores.append(score)
            except Exception as e:
                print(f"Error evaluating sample {sample.id}: {e}")
                continue

        category_scores = aggregate_by_category(samples, sample_scores)
        overall = calculate_overall_metrics(sample_scores)

        return EvalRunResult(
            run_id=run_id,
            dataset_name=self.dataset.name,
            variant_id=variant_id,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            sample_scores=sample_scores,
            category_scores=category_scores,
            total_samples=len(sample_scores),
            total_tp=overall["total_tp"],
            total_fp=overall["total_fp"],
            total_fn=overall["total_fn"],
            overall_precision=overall["precision"],
            overall_recall=overall["recall"],
            overall_f1=overall["f1"],
            predictions=predictions,
        )


# =============================================================================
# LangSmith Integration Helpers
# =============================================================================


def create_langsmith_dataset_examples(dataset: EvalDataset) -> list[dict]:
    """
    Convert EvalDataset to LangSmith dataset format.

    Returns list of examples for LangSmith dataset creation:
    [
        {
            "inputs": {"diff": "..."},
            "outputs": {"expected": {...}},
            "metadata": {...}
        },
        ...
    ]
    """
    examples = []
    for sample in dataset.samples:
        examples.append({
            "inputs": {
                "diff": sample.input.diff,
                "context_files": [
                    {"path": cf.path, "content": cf.content}
                    for cf in sample.input.context_files
                ],
            },
            "outputs": {
                "expected_issues": [
                    {
                        "category": issue.category.value,
                        "severity_min": issue.severity_min.value,
                        "title_keywords": issue.title_keywords,
                        "description_keywords": issue.description_keywords,
                        "issue_id": issue.issue_id,
                    }
                    for issue in sample.expected.issues
                ],
                "min_issues": sample.expected.min_issues,
                "max_issues": sample.expected.max_issues,
                "expected_risk": sample.expected.expected_risk.value if sample.expected.expected_risk else None,
                "should_have_blockers": sample.expected.should_have_blockers,
            },
            "metadata": {
                "sample_id": sample.id,
                "category": sample.metadata.primary_category.value,
                "difficulty": sample.metadata.difficulty.value,
                "tags": sample.metadata.tags,
            },
        })
    return examples


def create_langsmith_evaluator(
    scorer_fn: Optional[Callable] = None,
) -> Callable:
    """
    Create a LangSmith-compatible evaluator function.

    Usage with LangSmith:
        from langsmith import evaluate

        evaluator = create_langsmith_evaluator()

        results = evaluate(
            target=my_review_function,
            data=dataset_name,
            evaluators=[evaluator],
        )

    Args:
        scorer_fn: Optional custom scorer (defaults to score_sample)

    Returns:
        LangSmith-compatible evaluator function
    """
    def evaluator(run, example) -> dict:
        """
        LangSmith evaluator function.

        Args:
            run: LangSmith Run object with outputs
            example: LangSmith Example with inputs/outputs

        Returns:
            Dict with scores
        """
        from backend.evaluation.schemas import (
            ExpectedIssue,
            ExpectedResult,
            EvalSample,
            EvalInput,
            SampleMetadata,
        )
        from backend.domain.schemas.review import Category, Severity, RiskLevel

        # Reconstruct expected result from example
        expected_data = example.outputs
        expected_issues = [
            ExpectedIssue(
                category=Category(ei["category"]),
                severity_min=Severity(ei["severity_min"]),
                title_keywords=ei.get("title_keywords", []),
                description_keywords=ei.get("description_keywords", []),
                issue_id=ei.get("issue_id", ""),
            )
            for ei in expected_data.get("expected_issues", [])
        ]

        expected_risk = None
        if expected_data.get("expected_risk"):
            expected_risk = RiskLevel(expected_data["expected_risk"])

        expected = ExpectedResult(
            issues=expected_issues,
            min_issues=expected_data.get("min_issues", 0),
            max_issues=expected_data.get("max_issues"),
            expected_risk=expected_risk,
            should_have_blockers=expected_data.get("should_have_blockers"),
        )

        # Create minimal sample for scoring
        sample = EvalSample(
            id=example.metadata.get("sample_id", "unknown"),
            input=EvalInput(diff=example.inputs["diff"]),
            expected=expected,
            metadata=SampleMetadata(
                primary_category=Category(example.metadata.get("category", "correctness")),
            ),
        )

        # Get prediction from run
        prediction = run.outputs

        # If prediction is already ReviewResult, use it directly
        # Otherwise, try to parse it
        if not isinstance(prediction, ReviewResult):
            prediction = ReviewResult.model_validate(prediction)

        # Score
        score = score_sample(sample, prediction)

        return {
            "precision": score.precision,
            "recall": score.recall,
            "f1_score": score.f1_score,
            "true_positives": score.true_positives,
            "false_positives": score.false_positives,
            "false_negatives": score.false_negatives,
        }

    return evaluator
