"""
Scoring logic for evaluation.

Handles matching between expected issues and predicted issues,
and calculates precision/recall/F1 metrics.

Matching Strategy (Relaxed):
- Category + Severity are required (hard constraints)
- Keywords and semantic similarity are soft signals that contribute to a match score
- An issue matches if it captures the "essence" of the expected problem
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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
# Configuration
# =============================================================================

# Semantic matching thresholds
SEMANTIC_SIMILARITY_THRESHOLD = 0.4  # Minimum similarity to consider a match
KEYWORD_BOOST = 0.2  # Bonus score for keyword matches

# Whether to use semantic matching (can be disabled for faster tests)
USE_SEMANTIC_MATCHING = True


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
    details: dict = field(default_factory=dict)


def _keyword_match(text: str, keywords: list[str]) -> tuple[bool, float]:
    """
    Check if any keyword is found in text (case-insensitive).

    Args:
        text: Text to search in
        keywords: Keywords to search for (OR condition)

    Returns:
        Tuple of (any_matched, match_ratio)
    """
    if not keywords:
        return True, 1.0  # No keywords = no constraint

    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)

    return matches > 0, matches / len(keywords)


def _semantic_match(
    predicted_text: str,
    expected_keywords: list[str],
    rationale: str = "",
) -> tuple[bool, float]:
    """
    Check semantic similarity between predicted text and expected content.

    Args:
        predicted_text: The predicted issue title/description
        expected_keywords: Keywords that describe the expected issue
        rationale: Additional context about what should be found

    Returns:
        Tuple of (matched, similarity_score)
    """
    if not USE_SEMANTIC_MATCHING:
        return False, 0.0

    if not expected_keywords and not rationale:
        return True, 1.0

    try:
        from backend.evaluation.semantic_matcher import get_semantic_matcher
        matcher = get_semantic_matcher()

        # Build reference text from keywords and rationale
        reference_parts = []
        if expected_keywords:
            reference_parts.append(" ".join(expected_keywords))
        if rationale:
            reference_parts.append(rationale)

        reference_text = " ".join(reference_parts)

        if not reference_text.strip() or not predicted_text.strip():
            return False, 0.0

        similarity = matcher.similarity(predicted_text, reference_text)

        return similarity >= SEMANTIC_SIMILARITY_THRESHOLD, similarity

    except ImportError:
        # sentence-transformers not installed
        return False, 0.0


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

    Relaxed Matching Strategy:
    - Hard constraints (must pass):
      1. Category must match
      2. Severity must be >= minimum

    - Soft signals (contribute to match score):
      3. Keyword matching (title/description)
      4. Semantic similarity (title+description vs keywords+rationale)
      5. File pattern (if specified)
      6. Line number (if specified)

    An issue is considered a match if:
    - Hard constraints pass AND
    - (Keywords match OR semantic similarity >= threshold)

    Args:
        expected: Expected issue from ground truth
        predicted: Predicted issue from review

    Returns:
        MatchResult with matching details
    """
    details = {
        "category_match": False,
        "severity_match": False,
        "title_keyword_match": False,
        "title_keyword_ratio": 0.0,
        "desc_keyword_match": False,
        "desc_keyword_ratio": 0.0,
        "semantic_match": False,
        "semantic_score": 0.0,
        "file_match": False,
        "line_match": False,
        "final_score": 0.0,
    }

    # ===================
    # Hard constraints
    # ===================

    # 1. Category must match
    if predicted.category != expected.category:
        return MatchResult(matched=False, details=details)
    details["category_match"] = True

    # 2. Severity must be >= minimum
    if not severity_gte(predicted.severity, expected.severity_min):
        return MatchResult(matched=False, details=details)
    details["severity_match"] = True

    # ===================
    # Soft signals
    # ===================

    # 3. Keyword matching
    title_kw_match, title_kw_ratio = _keyword_match(
        predicted.title, expected.title_keywords
    )
    details["title_keyword_match"] = title_kw_match
    details["title_keyword_ratio"] = title_kw_ratio

    description_text = f"{predicted.title} {predicted.description} {predicted.why_it_matters}"
    desc_kw_match, desc_kw_ratio = _keyword_match(
        description_text, expected.description_keywords
    )
    details["desc_keyword_match"] = desc_kw_match
    details["desc_keyword_ratio"] = desc_kw_ratio

    # 4. Semantic similarity
    semantic_match, semantic_score = _semantic_match(
        predicted_text=description_text,
        expected_keywords=expected.title_keywords + expected.description_keywords,
        rationale=expected.rationale,
    )
    details["semantic_match"] = semantic_match
    details["semantic_score"] = semantic_score

    # 5. File pattern (if specified)
    file_matched = True
    if expected.file_pattern:
        file_matched = False
        for loc in predicted.locations:
            if _file_pattern_match(loc.file, expected.file_pattern):
                file_matched = True
                break
    details["file_match"] = file_matched

    # 6. Line number (if specified)
    line_matched = True
    if expected.line_start is not None:
        line_matched = False
        for loc in predicted.locations:
            if _line_match(loc.line_start, expected.line_start, expected.line_tolerance):
                line_matched = True
                break
    details["line_match"] = line_matched

    # ===================
    # Final decision
    # ===================

    # Calculate combined score
    # - Keyword matches contribute
    # - Semantic similarity contributes
    # - File/line matches are bonuses

    keyword_score = max(title_kw_ratio, desc_kw_ratio) if (title_kw_match or desc_kw_match) else 0.0

    # Combine keyword and semantic scores
    # Use max to allow either approach to succeed
    content_score = max(keyword_score, semantic_score)

    # Add bonuses for location matches
    if file_matched and expected.file_pattern:
        content_score += 0.1
    if line_matched and expected.line_start is not None:
        content_score += 0.1

    details["final_score"] = min(content_score, 1.0)

    # Match if:
    # - Any keyword matched, OR
    # - Semantic similarity is above threshold
    content_matched = title_kw_match or desc_kw_match or semantic_match

    if content_matched:
        return MatchResult(
            matched=True,
            matched_issue_id=predicted.id,
            match_score=details["final_score"],
            details=details,
        )

    return MatchResult(matched=False, match_score=0.0, details=details)


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
    best_match: Optional[tuple[Issue, MatchResult]] = None
    best_score = 0.0

    for pred in predictions:
        if pred.id in already_matched:
            continue

        result = match_issue(expected, pred)
        if result.matched and result.match_score > best_score:
            best_match = (pred, result)
            best_score = result.match_score

    return best_match


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
