"""Load regression thresholds from eval/thresholds.yaml, merged over defaults."""
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from rag_pipeline.eval.schema import MetricThreshold, Thresholds

DEFAULT_THRESHOLDS = Thresholds(
    metrics={
        "citation_precision": MetricThreshold(warn=0.02, fail=0.05),
        "citation_recall": MetricThreshold(warn=0.02, fail=0.05),
        "citation_f1": MetricThreshold(warn=0.02, fail=0.05),
        "coverage": MetricThreshold(warn=0.02, fail=0.05),
        "judge_score": MetricThreshold(warn=0.05, fail=0.10),
    },
    error_count=MetricThreshold(warn=0, fail=1),
    per_question_fail=0.5,
)

DEFAULT_THRESHOLDS_PATH = Path("eval/thresholds.yaml")


def load_thresholds(path: Optional[str | Path]) -> Thresholds:
    if path is None:
        return DEFAULT_THRESHOLDS
    path = Path(path)
    if not path.exists():
        return DEFAULT_THRESHOLDS
    try:
        raw = yaml.safe_load(path.read_text()) or {}
        section = (raw.get("evaluation") or {})
        merged_metrics = {name: t.model_copy() for name, t in DEFAULT_THRESHOLDS.metrics.items()}
        for name, tiers in (section.get("thresholds") or {}).items():
            merged_metrics[name] = MetricThreshold.model_validate(tiers)
        error_count = (
            MetricThreshold.model_validate(section["error_count"])
            if "error_count" in section
            else DEFAULT_THRESHOLDS.error_count
        )
        return Thresholds(
            metrics=merged_metrics,
            error_count=error_count,
            per_question_fail=section.get("per_question_fail", DEFAULT_THRESHOLDS.per_question_fail),
        )
    except (yaml.YAMLError, ValidationError, TypeError, AttributeError) as e:
        raise ValueError(f"Invalid thresholds file {path.name}: {e}") from e
