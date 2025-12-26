from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Severity(str, Enum):
    blocker = "blocker"
    high = "high"
    medium = "medium"
    low = "low"


class Category(str, Enum):
    correctness = "correctness"
    security = "security"
    performance = "performance"
    reliability = "reliability"
    api_compat = "api-compat"
    maintainability = "maintainability"
    style = "style"
    testing = "testing"
    docs = "docs"


class Meta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variant_id: str = Field(default="G0-baseline")
    run_id: str = Field(default="unknown")
    repair_used: bool = Field(default=False)
    llm_provider: str = Field(default="")
    model: str = Field(default="")
    repo: str = Field(default="")
    diff_target: str = Field(default="raw")
    generated_at: str = Field(default="")


class Location(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file: str
    line_start: int
    line_end: int


class Issue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    severity: Severity
    category: Category
    description: str
    why_it_matters: str = ""
    suggested_fix: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    locations: List[Location] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TestSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    type: str = Field(default="unit")
    target: str = ""
    rationale: str = ""
    evidence_ids: List[str] = Field(default_factory=list)


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    reason: str = ""
    evidence_ids: List[str] = Field(default_factory=list)


class PatchSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file: str
    unified_diff: str
    rationale: str = ""
    evidence_ids: List[str] = Field(default_factory=list)


class Summary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: str = ""
    overall_risk: RiskLevel = RiskLevel.low
    key_points: List[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    meta: Meta = Field(default_factory=Meta)
    summary: Summary = Field(default_factory=Summary)
    issues: List[Issue] = Field(default_factory=list)
    test_suggestions: List[TestSuggestion] = Field(default_factory=list)
    questions_to_author: List[Question] = Field(default_factory=list)
    merge_blockers: List[str] = Field(default_factory=list)
    patch_suggestions: List[PatchSuggestion] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    diff: Optional[str] = None
    diff_target: Optional[str] = Field(default="staged")
    variant_id: str = Field(default="g0-baseline")
