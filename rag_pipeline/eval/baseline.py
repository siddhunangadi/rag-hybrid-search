"""Baseline persistence: save/load eval/baselines/<name>.json. Validation via schema models."""
import hashlib
import json
import os
from pathlib import Path

from pydantic import ValidationError

from rag_pipeline.eval.schema import BASELINE_VERSION, Baseline

DEFAULT_BASE_DIR = Path("eval/baselines")


class BaselineMissingError(FileNotFoundError):
    pass


class BaselineCorruptError(ValueError):
    pass


def question_set_hash(questions_path: str | Path) -> str:
    return hashlib.sha256(Path(questions_path).read_bytes()).hexdigest()


def baseline_path(name: str, base_dir: str | Path = DEFAULT_BASE_DIR) -> Path:
    return Path(base_dir) / f"{name}.json"


def save_baseline(baseline: Baseline, name: str, base_dir: str | Path = DEFAULT_BASE_DIR) -> Path:
    path = baseline_path(name, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: a crash mid-write must never leave a truncated baseline
    # that later fails as "corrupt". Same-directory temp file so os.replace
    # stays on one filesystem (rename is atomic only within a filesystem).
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        f.write(baseline.model_dump_json(indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    return path


def load_baseline(name: str, base_dir: str | Path = DEFAULT_BASE_DIR) -> Baseline:
    path = baseline_path(name, base_dir)
    if not path.exists():
        raise BaselineMissingError(
            f"No baseline at {path}. Create one with: python scripts/run_eval.py --update-baseline"
        )
    try:
        payload = json.loads(path.read_text())
        baseline = Baseline.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as e:
        raise BaselineCorruptError(f"Corrupt baseline {path}: {e}") from e
    if baseline.baseline_version != BASELINE_VERSION:
        raise BaselineCorruptError(
            f"Unsupported baseline_version {baseline.baseline_version} in {path} "
            f"(supported: {BASELINE_VERSION})"
        )
    return baseline
