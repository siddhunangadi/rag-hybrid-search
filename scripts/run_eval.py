#!/usr/bin/env python3
import argparse
import json
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

from rag_hybrid_search.trace import RequestTrace
from rag_pipeline.eval.metrics import error_record, evaluate_question
from rag_pipeline.eval.questions import load_questions
from rag_pipeline.eval.report import write_report
from rag_pipeline.generation_provider import MockProvider

_SECRET_SETTINGS_FIELDS = {"nvidia_api_key", "gemini_api_key", "debug_token"}


def _package_version() -> str:
    try:
        return version("rag-hybrid-search")
    except PackageNotFoundError:
        return "unknown"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--fixture-pipeline", action="store_true", help="Use an in-memory fixture pipeline instead of building the real one (for smoke-testing this script).")
    args = parser.parse_args()

    dataset, questions = load_questions(args.questions)
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
    }

    out_dir = args.out_dir or f"eval/reports/{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}"
    json_path, html_path = write_report(records, metadata, out_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
