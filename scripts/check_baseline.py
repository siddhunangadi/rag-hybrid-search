#!/usr/bin/env python3
# scripts/check_baseline.py
"""Compare an existing report.json against a stored baseline without re-running eval."""
import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag_pipeline.eval.baseline import (  # noqa: E402
    BaselineCorruptError,
    BaselineMissingError,
    load_baseline,
)
from rag_pipeline.eval.comparison import QuestionSetMismatchError, compare  # noqa: E402
from rag_pipeline.eval.renderer import render_comparison  # noqa: E402
from rag_pipeline.eval.snapshot import snapshot_from_records  # noqa: E402
from rag_pipeline.eval.thresholds import load_thresholds  # noqa: E402
from scripts.run_eval import (  # noqa: E402
    EXIT_BASELINE_CORRUPT,
    EXIT_BASELINE_MISSING,
    EXIT_HASH_MISMATCH,
    EXIT_REGRESSION,
    EXIT_USAGE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--baseline-name", default="main")
    parser.add_argument("--baseline-dir", default="eval/baselines")
    parser.add_argument("--thresholds", default="eval/thresholds.yaml")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    q_hash = report.get("metadata", {}).get("question_set_hash")
    if not q_hash:
        print(
            "Report has no metadata.question_set_hash (pre-Phase 2 report?). "
            "Re-run scripts/run_eval.py to produce a comparable report.",
            file=sys.stderr,
        )
        sys.exit(EXIT_USAGE)

    snapshot = snapshot_from_records(
        report["results"], report["summary"], q_hash,
        report.get("metadata", {}).get("pipeline_config", {}),
    )
    try:
        baseline = load_baseline(args.baseline_name, base_dir=args.baseline_dir)
        result = compare(snapshot, baseline.snapshot(), load_thresholds(args.thresholds))
    except BaselineMissingError as e:
        print(str(e), file=sys.stderr)
        sys.exit(EXIT_BASELINE_MISSING)
    except BaselineCorruptError as e:
        print(str(e), file=sys.stderr)
        sys.exit(EXIT_BASELINE_CORRUPT)
    except QuestionSetMismatchError as e:
        print(str(e), file=sys.stderr)
        sys.exit(EXIT_HASH_MISMATCH)

    print(render_comparison(result))
    if result.overall == "fail":
        sys.exit(EXIT_REGRESSION)


if __name__ == "__main__":
    main()
