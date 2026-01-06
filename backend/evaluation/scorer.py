"""
Scoring logic for evaluation.

Handles matching between expected issues and predicted issues,
and calculates precision/recall/F1 metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from backend.domain.schemas.review import (
    ReviewResult,
    Issue,
    Category,
    Severity,
)
from backend.evaluation.schemas import (
    EvalSample,
    ExpectedIssue,
    ExpectedResult,
    IssueMatch,
    SampleScore,
    CategoryScore,
    EvalRunResult,
)


# =============================================================================
# Severity ordering for comparison
# =============================================================================

SEVERITY_ORDER = {
    Severity.low: 0,
    Severity.medium: 1,
    Severity.high: 2,
    Severity.blocker: 3,
}


def severity_gte(actual: Severity, minimum: Severity) -> bool:
    """Check if actual severity is >= minimum severity."""
    return SEVERITY_ORDER[actual] >= SEVERITY_ORDER[minimum]


# =============================================================================
# Issue Matching
# =============================================================================


@dataclass
class MatchResult:
    """Result of matching a single expected issue against predictions."""
    matched: bool
    matched_issue_id: Optional[str] = None
    match_score: float = 0.0
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


def _keyword_match(text: str, keywords: list[str]) -> bool:
    """
    Check if any keyword is found in text (case-insensitive).

    Args:
        text: Text to search in
        keywords: Keywords to search for (OR condition)

    Returns:
        True if any keyword is found
    """
    if not keywords:
        return True  # No keywords = no constraint

    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _file_pattern_match(file_path: str, pattern: Optional[str]) -> bool:
    """
    Check if file path matches pattern.

    Args:
        file_path: Actual file path
        pattern: Glob-like pattern or substring

    Returns:
        True if matches or no pattern specified
    """
    if not pattern:
        return True

    # Simple substring match for now
    # Could be extended to support glob patterns
    return pattern.lower() in file_path.lower()


def _line_match(
    actual_start: int,
    expected_start: Optional[int],
    tolerance: int = 3,
) -> bool:
    """
    Check if line numbers are within tolerance.

    Args:
        actual_start: Actual line start
        expected_start: Expected line start (if specified)
        tolerance: Allowed deviation

    Returns:
        True if within tolerance or no expected specified
    """
    if expected_start is None:
        return True

    return abs(actual_start - expected_start) <= tolerance


def match_issue(
    expected: ExpectedIssue,
    predicted: Issue,
) -> MatchResult:
    """
    Check if a predicted issue matches an expected issue.

    Matching criteria (all must pass):
    1. Category must match
    2. Severity must be >= minimum
    3. Title keywords (OR - at least one must match)
    4. Description keywords (OR - at least one must match)
    5. File pattern (if specified)
    6. Line number (if specified, with tolerance)

    Args:
        expected: Expected issue from ground truth
        predicted: Predicted issue from review

    Returns:
        MatchResult with matching details
    """
    details = {
        "category_match": False,
        "severity_match": False,
        "title_keywords_match": False,
        "description_keywords_match": False,
        "file_match": False,
        "line_match": False,
    }

    # 1. Category must match
    if predicted.category != expected.category:
        return MatchResult(matched=False, details=details)
    details["category_match"] = True

    # 2. Severity must be >= minimum
    if not severity_gte(predicted.severity, expected.severity_min):
        return MatchResult(matched=False, details=details)
    details["severity_match"] = True

    # 3. Title keywords (OR condition)
    if not _keyword_match(predicted.title, expected.title_keywords):
        return MatchResult(matched=False, details=details)
    details["title_keywords_match"] = True

    # 4. Description keywords (OR condition)
    description_text = f"{predicted.description} {predicted.why_it_matters}"
    if not _keyword_match(description_text, expected.description_keywords):
        return MatchResult(matched=False, details=details)
    details["description_keywords_match"] = True

    # 5. File pattern (if specified)
    file_matched = False
    if expected.file_pattern:
        for loc in predicted.locations:
            if _file_pattern_match(loc.file, expected.file_pattern):
                file_matched = True
                break
        if not file_matched and predicted.locations:
            return MatchResult(matched=False, details=details)
    else:
        file_matched = True
    details["file_match"] = file_matched

    # 6. Line number (if specified)
    line_matched = False
    if expected.line_start is not None:
        for loc in predicted.locations:
            if _line_match(loc.line_start, expected.line_start, expected.line_tolerance):
                line_matched = True
                break
        if not line_matched and predicted.locations:
            return MatchResult(matched=False, details=details)
    else:
        line_matched = True
    details["line_match"] = line_matched

    # All criteria passed
    return MatchResult(
        matched=True,
        matched_issue_id=predicted.id,
        match_score=1.0,
        details=details,
    )


def find_best_match(
    expected: ExpectedIssue,
    predictions: list[Issue],
    already_matched: set[str],
) -> Optional[tuple[Issue, MatchResult]]:
    """
    Find the best matching predicted issue for an expected issue.

    Args:
        expected: Expected issue to match
        predictions: List of predicted issues
        already_matched: Set of already matched prediction IDs

    Returns:
        Tuple of (matched issue, match result) or None if no match
    """
    for pred in predictions:
        if pred.id in already_matched:
            continue

        result = match_issue(expected, pred)
        if result.matched:
            return pred, result

    return None


# =============================================================================
# Sample Scoring
# =============================================================================


def score_sample(
    sample: EvalSample,
    prediction: ReviewResult,
) -> SampleScore:
    """
    Score a single sample by comparing prediction to expected result.

    Args:
        sample: Evaluation sample with ground truth
        prediction: Review result from the model

    Returns:
        SampleScore with metrics
    """
    expected = sample.expected
    predicted_issues = prediction.issues

    # Track which predictions have been matched
    matched_prediction_ids: set[str] = set()
    issue_matches: list[IssueMatch] = []

    # Match each expected issue
    for exp_issue in expected.issues:
        match_result = find_best_match(
            exp_issue,
            predicted_issues,
            matched_prediction_ids,
        )

        if match_result:
            pred, result = match_result
            matched_prediction_ids.add(pred.id)
            issue_matches.append(IssueMatch(
                expected_id=exp_issue.issue_id,
                predicted_id=pred.id,
                matched=True,
                match_details=result.details,
            ))
        else:
            issue_matches.append(IssueMatch(
                expected_id=exp_issue.issue_id,
                predicted_id=None,
                matched=False,
                match_details={},
            ))

    # Find unmatched predictions (potential FPs)
    unmatched_predictions = [
        p.id for p in predicted_issues
        if p.id not in matched_prediction_ids
    ]

    # Calculate counts
    true_positives = sum(1 for m in issue_matches if m.matched)
    false_negatives = sum(1 for m in issue_matches if not m.matched)
    false_positives = len(unmatched_predictions)

    # Check forbidden categories
    forbidden_violated = False
    if expected.forbidden_categories:
        for pred in predicted_issues:
            if pred.category in expected.forbidden_categories:
                forbidden_violated = True
                break

    # Check risk level
    risk_correct = None
    if expected.expected_risk is not None:
        risk_correct = prediction.summary.overall_risk == expected.expected_risk

    # Check blockers
    blocker_correct = None
    if expected.should_have_blockers is not None:
        has_blockers = len(prediction.merge_blockers) > 0
        blocker_correct = has_blockers == expected.should_have_blockers

    # Calculate metrics
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    return SampleScore(
        sample_id=sample.id,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        issue_matches=issue_matches,
        unmatched_predictions=unmatched_predictions,
        risk_level_correct=risk_correct,
        blocker_correct=blocker_correct,
        forbidden_category_violated=forbidden_violated,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )


# =============================================================================
# Aggregate Scoring
# =============================================================================


def aggregate_by_category(
    samples: list[EvalSample],
    scores: list[SampleScore],
) -> list[CategoryScore]:
    """
    Aggregate scores by category.

    Args:
        samples: List of evaluation samples
        scores: Corresponding list of sample scores

    Returns:
        List of CategoryScore, one per category
    """
    # Build sample_id -> sample mapping
    sample_map = {s.id: s for s in samples}

    # Group by category
    category_data: dict[Category, dict] = {}

    for score in scores:
        sample = sample_map.get(score.sample_id)
        if not sample:
            continue

        cat = sample.metadata.primary_category

        if cat not in category_data:
            category_data[cat] = {
                "sample_count": 0,
                "total_tp": 0,
                "total_fp": 0,
                "total_fn": 0,
            }

        category_data[cat]["sample_count"] += 1
        category_data[cat]["total_tp"] += score.true_positives
        category_data[cat]["total_fp"] += score.false_positives
        category_data[cat]["total_fn"] += score.false_negatives

    # Calculate metrics per category
    result = []
    for cat, data in category_data.items():
        tp = data["total_tp"]
        fp = data["total_fp"]
        fn = data["total_fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        result.append(CategoryScore(
            category=cat,
            sample_count=data["sample_count"],
            total_tp=tp,
            total_fp=fp,
            total_fn=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
        ))

    return result


def calculate_overall_metrics(scores: list[SampleScore]) -> dict:
    """
    Calculate overall metrics across all samples.

    Args:
        scores: List of sample scores

    Returns:
        Dict with total_tp, total_fp, total_fn, precision, recall, f1
    """
    total_tp = sum(s.true_positives for s in scores)
    total_fp = sum(s.false_positives for s in scores)
    total_fn = sum(s.false_negatives for s in scores)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
