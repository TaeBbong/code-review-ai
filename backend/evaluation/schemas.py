"""
Evaluation schemas for code review bot performance measurement.

These schemas define the structure for:
- Evaluation datasets (input + expected output pairs)
- Evaluation results (predicted vs ground truth comparison)
- Scoring metrics
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from backend.domain.schemas.review import (
    Category,
    Severity,
    RiskLevel,
    ReviewResult,
)


# =============================================================================
# Enums
# =============================================================================


class Difficulty(str, Enum):
    """Difficulty level of the evaluation sample."""
    easy = "easy"
    medium = "medium"
    hard = "hard"


class DataSource(str, Enum):
    """Source of the evaluation sample."""
    synthetic = "synthetic"      # 인위적으로 생성
    real_world = "real_world"    # 실제 PR에서 수집
    cve_based = "cve_based"      # CVE/보안 DB 기반
    humanevalfix = "humanevalfix"  # HumanEvalFix 데이터셋


# =============================================================================
# Input Schemas
# =============================================================================


class ContextFile(BaseModel):
    """Additional context file for evidence collection."""
    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., description="파일 경로")
    content: str = Field(..., description="파일 내용")


class EvalInput(BaseModel):
    """Input data for evaluation - the diff to be reviewed."""
    model_config = ConfigDict(extra="forbid")

    diff: str = Field(..., description="리뷰할 diff 내용")
    context_files: List[ContextFile] = Field(
        default_factory=list,
        description="Evidence 수집용 추가 컨텍스트 파일"
    )


# =============================================================================
# Expected Output Schemas (Ground Truth)
# =============================================================================


class ExpectedIssue(BaseModel):
    """Expected issue that should be detected."""
    model_config = ConfigDict(extra="forbid")

    # 매칭 기준
    category: Category = Field(..., description="이슈 카테고리")
    severity_min: Severity = Field(
        default=Severity.low,
        description="최소 심각도 (이 이상이어야 함)"
    )

    # 위치 매칭 (선택적)
    file_pattern: Optional[str] = Field(
        default=None,
        description="파일 경로 패턴 (glob 또는 substring)"
    )
    line_start: Optional[int] = Field(
        default=None,
        description="예상 시작 라인 (±tolerance 허용)"
    )
    line_tolerance: int = Field(
        default=3,
        description="라인 매칭 허용 오차"
    )

    # 내용 매칭
    title_keywords: List[str] = Field(
        default_factory=list,
        description="제목에 포함되어야 할 키워드 (OR 조건)"
    )
    description_keywords: List[str] = Field(
        default_factory=list,
        description="설명에 포함되어야 할 키워드 (OR 조건)"
    )

    # 메타데이터
    issue_id: str = Field(
        default="",
        description="Ground truth 이슈 식별자 (매칭 추적용)"
    )
    rationale: str = Field(
        default="",
        description="왜 이 이슈가 발견되어야 하는지 설명"
    )


class ExpectedResult(BaseModel):
    """Ground truth - what the review should find."""
    model_config = ConfigDict(extra="forbid")

    # 필수 발견 이슈
    issues: List[ExpectedIssue] = Field(
        default_factory=list,
        description="반드시 발견해야 할 이슈 목록"
    )

    # 이슈 개수 제약
    min_issues: int = Field(
        default=0,
        description="최소 이슈 개수"
    )
    max_issues: Optional[int] = Field(
        default=None,
        description="최대 이슈 개수 (초과시 과탐 의심)"
    )

    # 전체 리스크 레벨
    expected_risk: Optional[RiskLevel] = Field(
        default=None,
        description="예상 전체 리스크 레벨"
    )

    # 머지 블로커 여부
    should_have_blockers: Optional[bool] = Field(
        default=None,
        description="머지 블로커가 있어야 하는지"
    )

    # False Positive 체크용: 이런 이슈는 없어야 함
    forbidden_categories: List[Category] = Field(
        default_factory=list,
        description="발견되면 안 되는 카테고리 (FP 체크)"
    )


# =============================================================================
# Sample Metadata
# =============================================================================


class SampleMetadata(BaseModel):
    """Metadata about the evaluation sample."""
    model_config = ConfigDict(extra="forbid")

    source: DataSource = Field(
        default=DataSource.synthetic,
        description="데이터 출처"
    )
    difficulty: Difficulty = Field(
        default=Difficulty.medium,
        description="난이도"
    )
    primary_category: Category = Field(
        ...,
        description="주요 테스트 카테고리"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="추가 태그 (예: owasp-top10, n-plus-one)"
    )
    description: str = Field(
        default="",
        description="샘플에 대한 설명"
    )
    created_at: str = Field(
        default="",
        description="생성 일시"
    )
    author: str = Field(
        default="",
        description="작성자"
    )


# =============================================================================
# Evaluation Sample (Input + Expected)
# =============================================================================


class EvalSample(BaseModel):
    """Single evaluation sample with input and expected output."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="샘플 고유 ID (예: correctness-001)")
    input: EvalInput = Field(..., description="입력 데이터")
    expected: ExpectedResult = Field(..., description="기대 결과 (Ground Truth)")
    metadata: SampleMetadata = Field(..., description="샘플 메타데이터")


class EvalDataset(BaseModel):
    """Collection of evaluation samples."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="데이터셋 이름")
    version: str = Field(default="1.0.0", description="데이터셋 버전")
    description: str = Field(default="", description="데이터셋 설명")
    samples: List[EvalSample] = Field(default_factory=list, description="샘플 목록")


# =============================================================================
# Evaluation Results
# =============================================================================


class IssueMatch(BaseModel):
    """Matching result between expected and predicted issue."""
    model_config = ConfigDict(extra="forbid")

    expected_id: str = Field(..., description="Expected issue ID")
    predicted_id: Optional[str] = Field(
        default=None,
        description="매칭된 predicted issue ID (없으면 FN)"
    )
    matched: bool = Field(default=False, description="매칭 성공 여부")
    match_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="매칭 상세 정보 (어떤 기준으로 매칭되었는지)"
    )


class SampleScore(BaseModel):
    """Scores for a single evaluation sample."""
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(..., description="샘플 ID")

    # 기본 카운트
    true_positives: int = Field(default=0, description="정탐 (TP)")
    false_positives: int = Field(default=0, description="오탐 (FP)")
    false_negatives: int = Field(default=0, description="미탐 (FN)")

    # 매칭 상세
    issue_matches: List[IssueMatch] = Field(
        default_factory=list,
        description="이슈별 매칭 결과"
    )
    unmatched_predictions: List[str] = Field(
        default_factory=list,
        description="매칭되지 않은 predicted issue IDs (FP 후보)"
    )

    # 추가 체크
    risk_level_correct: Optional[bool] = Field(
        default=None,
        description="리스크 레벨 일치 여부"
    )
    blocker_correct: Optional[bool] = Field(
        default=None,
        description="블로커 존재 여부 일치"
    )
    forbidden_category_violated: bool = Field(
        default=False,
        description="금지된 카테고리 이슈 발견 여부"
    )

    # 계산된 메트릭
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    f1_score: float = Field(default=0.0, ge=0.0, le=1.0)


class CategoryScore(BaseModel):
    """Aggregated scores per category."""
    model_config = ConfigDict(extra="forbid")

    category: Category
    sample_count: int = 0
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


class EvalRunResult(BaseModel):
    """Complete evaluation run result."""
    model_config = ConfigDict(extra="forbid")

    # 실행 정보
    run_id: str = Field(..., description="평가 실행 ID")
    dataset_name: str = Field(..., description="사용된 데이터셋")
    variant_id: str = Field(..., description="평가된 variant")
    evaluated_at: str = Field(..., description="평가 실행 시간")

    # 샘플별 결과
    sample_scores: List[SampleScore] = Field(
        default_factory=list,
        description="샘플별 점수"
    )

    # 카테고리별 집계
    category_scores: List[CategoryScore] = Field(
        default_factory=list,
        description="카테고리별 집계 점수"
    )

    # 전체 집계
    total_samples: int = 0
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1: float = 0.0

    # 원본 데이터 (디버깅/분석용)
    predictions: Dict[str, ReviewResult] = Field(
        default_factory=dict,
        description="샘플 ID -> ReviewResult 매핑"
    )
