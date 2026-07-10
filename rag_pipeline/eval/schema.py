"""Typed models for baseline storage and regression comparison (eval Phase 2).

All models are frozen: snapshots and baselines are immutable records of a run.
"""
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict

BASELINE_VERSION = 1

FindingStatus = Literal["ok", "warn", "fail", "info"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class MetricThreshold(_Frozen):
    warn: float
    fail: float


class Thresholds(_Frozen):
    metrics: dict[str, MetricThreshold]
    error_count: MetricThreshold
    per_question_fail: float = 0.5


class QuestionMetrics(_Frozen):
    status: str
    objective_metrics: dict[str, Optional[Union[float, bool]]] = {}


class SnapshotSummary(_Frozen):
    objective: Optional[dict[str, float]] = None
    subjective: Optional[dict[str, float]] = None
    error_count: int
    total_questions: int


class EvaluationSnapshot(_Frozen):
    question_set_hash: str
    summary: SnapshotSummary
    per_question: dict[str, QuestionMetrics]
    pipeline_config: dict = {}


class Baseline(_Frozen):
    baseline_version: int
    created_at: str
    git_commit: str = "unknown"
    package_version: str = "unknown"
    branch: Optional[str] = None
    notes: Optional[str] = None
    python_version: Optional[str] = None
    platform: Optional[str] = None
    question_set_hash: str
    pipeline_config: dict = {}
    summary: SnapshotSummary
    per_question: dict[str, QuestionMetrics]

    def snapshot(self) -> EvaluationSnapshot:
        return EvaluationSnapshot(
            question_set_hash=self.question_set_hash,
            summary=self.summary,
            per_question=self.per_question,
            pipeline_config=self.pipeline_config,
        )


class Finding(_Frozen):
    metric: str
    scope: str
    baseline: Optional[float] = None
    current: Optional[float] = None
    delta: Optional[float] = None
    status: FindingStatus


_SEVERITY = {"ok": 0, "info": 0, "warn": 1, "fail": 2}
_BY_SEVERITY = {0: "ok", 1: "warn", 2: "fail"}


class ComparisonResult(_Frozen):
    findings: list[Finding]

    @property
    def overall(self) -> str:
        worst = max((_SEVERITY[f.status] for f in self.findings), default=0)
        return _BY_SEVERITY[worst]
