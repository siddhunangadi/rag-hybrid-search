#!/usr/bin/env python3
import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Ensure the repo root is on sys.path when this script is invoked directly
# (e.g. `python scripts/run_eval.py`), since Python otherwise prepends the
# script's own directory rather than the cwd -- api.dependencies imports
# `tests.fakes` for its dev-fallback embedding provider, which needs the
# repo root importable.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag_hybrid_search.trace import RequestTrace  # noqa: E402
from rag_pipeline.eval.baseline import (  # noqa: E402
    BaselineCorruptError,
    BaselineMissingError,
    load_baseline,
    question_set_hash,
    save_baseline,
)
from rag_pipeline.eval.comparison import QuestionSetMismatchError, compare  # noqa: E402
from rag_pipeline.eval.metrics import error_record, evaluate_question  # noqa: E402
from rag_pipeline.eval.questions import load_questions  # noqa: E402
from rag_pipeline.eval.renderer import render_comparison  # noqa: E402
from rag_pipeline.eval.report import build_summary, write_report  # noqa: E402
from rag_pipeline.eval.schema import BASELINE_VERSION, Baseline  # noqa: E402
from rag_pipeline.eval.snapshot import snapshot_from_records  # noqa: E402
from rag_pipeline.eval.thresholds import load_thresholds  # noqa: E402
from rag_pipeline.generation_provider import MockProvider  # noqa: E402

_SECRET_SETTINGS_FIELDS = {"nvidia_api_key", "gemini_api_key", "debug_token"}


def _package_version() -> str:
    try:
        return version("rag-hybrid-search")
    except PackageNotFoundError:
        return "unknown"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_commit() -> str:
    return _git("rev-parse", "HEAD")


def _sanitized_settings(settings) -> dict:
    dumped = settings.model_dump()
    return {k: v for k, v in dumped.items() if k not in _SECRET_SETTINGS_FIELDS}


def _build_pipeline(use_fixture: bool):
    """Real pipeline via the app's container, or an in-memory fixture
    pipeline for the smoke test (--fixture-pipeline avoids requiring a
    populated data/ dir and a real or mock-provider network round trip
    in CI)."""
    if use_fixture:
        from rag_hybrid_search.config import Settings
        from rag_hybrid_search.models import Chunk, RetrievalTrace, RetrievedChunk
        from rag_pipeline.rag_pipeline import RagPipeline

        class _FixtureRetriever:
            def retrieve(self, query, dev_trace=None):
                chunk = Chunk(
                    chunk_id="c1", document_id="d1", chunk_index=0, text="X is a thing.",
                    strategy_version="fixed-v1", heading=None, page=None, char_count=13,
                )
                return [RetrievedChunk(
                    chunk=chunk, dense_score=0.9, bm25_score=0.9, rrf_score=0.9,
                    rerank_score=0.9, final_rank=1,
                )], RetrievalTrace()

        provider = MockProvider(canned_json=json.dumps({
            "answer": "X is a thing [d1].",
            "claims": [{"text": "X is a thing.", "citation_ids": ["d1"], "supporting_quote": "X is a thing."}],
        }))
        pipeline = RagPipeline(_FixtureRetriever(), provider)
        return pipeline, provider, Settings()

    from api.dependencies import build_container
    container = build_container()
    return container.rag_pipeline, container.generation_provider, container.settings


_PIPELINE_CONFIG_FIELDS = [
    "provider", "generation_model", "chunking_strategy", "chunk_size",
    "chunk_overlap", "dense_k", "sparse_k", "rerank_top_n", "rerank_backend",
]

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_BASELINE_MISSING = 2
EXIT_BASELINE_CORRUPT = 3
EXIT_HASH_MISMATCH = 4
EXIT_USAGE = 5


def _pipeline_config(settings) -> dict:
    return {f: getattr(settings, f, None) for f in _PIPELINE_CONFIG_FIELDS}


def _git_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--fixture-pipeline", action="store_true", help="Use an in-memory fixture pipeline instead of building the real one (for smoke-testing this script).")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--baseline-name", default="main")
    parser.add_argument("--baseline-dir", default="eval/baselines")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--thresholds", default="eval/thresholds.yaml")
    args = parser.parse_args()

    if args.update_baseline and args.compare_baseline:
        print("Use either --update-baseline or --compare-baseline, not both.", file=sys.stderr)
        sys.exit(EXIT_USAGE)

    dataset, questions = load_questions(args.questions)
    q_hash = question_set_hash(args.questions)
    pipeline, generation_provider, settings = _build_pipeline(args.fixture_pipeline)
    judge_provider = generation_provider  # Phase 1 default: judge = generation

    records = []
    for question in questions:
        trace = RequestTrace(question.question, {"Generation": type(generation_provider).__name__})
        started = time.perf_counter()
        try:
            rag_answer = pipeline.answer(question.question, dev_trace=trace)
            latency_ms = (time.perf_counter() - started) * 1000
            records.append(evaluate_question(question, rag_answer, trace.data, latency_ms, judge_provider))
        except Exception as e:
            records.append(error_record(question, error_type=type(e).__name__, error_message=str(e)))

    metadata = {
        "report_version": "1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "package_version": _package_version(),
        "generation_model": type(generation_provider).__name__,
        "judge_model": type(judge_provider).__name__,
        "prompt_version": getattr(pipeline, "prompt_version", "unknown"),
        "judge_prompt_version": "v1",
        "settings": _sanitized_settings(settings),
        "corpus_version": "unknown",
        "dataset": {"name": dataset.name, "version": dataset.version},
        "question_set_hash": q_hash,
        "pipeline_config": _pipeline_config(settings),
    }

    out_dir = args.out_dir or f"eval/reports/{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}"
    json_path, html_path = write_report(records, metadata, out_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {html_path}")

    if args.update_baseline or args.compare_baseline:
        snapshot = snapshot_from_records(
            records, build_summary(records), q_hash, _pipeline_config(settings)
        )

    if args.update_baseline:
        baseline = Baseline(
            baseline_version=BASELINE_VERSION,
            created_at=metadata["timestamp"],
            git_commit=metadata["git_commit"],
            package_version=metadata["package_version"],
            branch=_git_branch(),
            notes=args.notes,
            python_version=platform.python_version(),
            platform=platform.platform(),
            question_set_hash=q_hash,
            pipeline_config=snapshot.pipeline_config,
            summary=snapshot.summary,
            per_question=snapshot.per_question,
        )
        path = save_baseline(baseline, args.baseline_name, base_dir=args.baseline_dir)
        print(f"Wrote baseline {path}")

    if args.compare_baseline:
        try:
            baseline = load_baseline(args.baseline_name, base_dir=args.baseline_dir)
            thresholds = load_thresholds(args.thresholds)
            result = compare(snapshot, baseline.snapshot(), thresholds)
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
