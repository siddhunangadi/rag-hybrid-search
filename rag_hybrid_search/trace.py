"""Developer Trace Mode: prints every pipeline stage to stdout and dumps a
per-request JSON trace file, gated by the ``TRACE_RAG`` env var.

Off by default (near-zero cost: one env lookup per request). When enabled,
a ``RequestTrace`` is created once per ``/answer`` call, threaded through
retrieval and generation, and prints each stage as it happens so a request
can be watched live in the server log -- then the same data is written to
``traces/<timestamp>-<request_id>.json`` for later inspection.
"""

import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_BAR = "=" * 70

_EXECUTION_GRAPH = """
Question
   |
   v
Query Embedding
   |
   v
Dense Search -----+
                   |
BM25 Search -------+
                   v
             RRF Fusion
                   |
                   v
              Reranker
                   |
                   v
            Prompt Builder
                   |
                   v
             LLM Generate
                   |
                   v
        JSON Parse & Repair
                   |
                   v
      Citation Verification
                   |
                   v
        Confidence Scoring
                   |
                   v
             Final Answer
"""


def trace_enabled() -> bool:
    return os.environ.get("TRACE_RAG", "").strip().lower() in ("1", "true", "yes")


def _section(title: str) -> None:
    print(f"\n{_BAR}\n{title}\n{_BAR}")


def _kv(**kwargs) -> None:
    for k, v in kwargs.items():
        print(f"{k:<18}: {v}")


class RequestTrace:
    def __init__(self, question: str, config: dict, traces_dir: str = "traces"):
        self.enabled = trace_enabled()
        self.request_id = uuid4().hex[:8]
        self.question = question
        self.started_at = datetime.now(timezone.utc)
        self._t0 = time.perf_counter()
        self._timings: dict[str, float] = {}
        self._traces_dir = Path(traces_dir)
        self._data: dict = {
            "request_id": self.request_id,
            "question": question,
            "config": config,
            "timestamp": self.started_at.isoformat(),
        }
        if self.enabled:
            _section("REQUEST START")
            _kv(**{"Request ID": self.request_id, "Timestamp": self.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")}, **config)
            print(f"\nQuestion           : {question}")
            _kv(Characters=len(question), Words=len(question.split()))

    def mark(self, name: str, elapsed_ms: float) -> None:
        self._timings[name] = elapsed_ms

    def log_query_embedding(self, provider: str, model: str, dim: int, vector: list[float], latency_ms: float) -> None:
        norm = sum(v * v for v in vector) ** 0.5
        self._data["query_embedding"] = {
            "provider": provider, "model": model, "dim": dim, "norm": norm, "latency_ms": latency_ms,
        }
        self.mark("embedding", latency_ms)
        if not self.enabled:
            return
        _section("STEP 1 -- QUERY EMBEDDING")
        _kv(Provider=provider, Model=model, Dimension=dim, **{"Vector Norm": f"{norm:.2f}", "Latency": f"{latency_ms:.1f} ms"})
        preview = ", ".join(f"{v:.3f}" for v in vector[:5])
        print(f"Vector Preview    : [{preview}, ...]")

    def log_dense(self, results, latency_ms: float) -> None:
        self._data["dense"] = [
            {"chunk_id": r.chunk.chunk_id, "score": r.dense_score, "document_id": r.chunk.document_id,
             "page": r.chunk.page, "chars": r.chunk.char_count, "preview": r.chunk.text[:120]}
            for r in results
        ]
        self.mark("dense_search", latency_ms)
        if not self.enabled:
            return
        _section("STEP 2 -- DENSE VECTOR SEARCH")
        _kv(Returned=f"{len(results)} chunks", Latency=f"{latency_ms:.1f} ms")
        for rank, r in enumerate(results, 1):
            score = "n/a" if r.dense_score is None else f"{r.dense_score:.4f}"
            print(f"\n  Rank {rank}  chunk={r.chunk.chunk_id[:12]}  score={score}  page={r.chunk.page}  chars={r.chunk.char_count}")
            print(f"    {r.chunk.text[:120].strip()!r}")

    def log_bm25(self, results: list[tuple[str, float]], latency_ms: float) -> None:
        self._data["bm25"] = [{"chunk_id": cid, "score": score} for cid, score in results]
        self.mark("bm25_search", latency_ms)
        if not self.enabled:
            return
        _section("STEP 3 -- BM25 KEYWORD SEARCH")
        _kv(Returned=f"{len(results)} chunks", Latency=f"{latency_ms:.1f} ms")
        for rank, (cid, score) in enumerate(results, 1):
            print(f"  Rank {rank}  chunk={cid[:12]}  score={score:.3f}")

    def log_fusion(
        self, fused: list, dense_ids_ranked: list[str], bm25_ids_ranked: list[str],
        rrf_k: int, dense_weight: float, sparse_weight: float, latency_ms: float,
    ) -> None:
        dense_rank = {cid: i + 1 for i, cid in enumerate(dense_ids_ranked)}
        bm25_rank = {cid: i + 1 for i, cid in enumerate(bm25_ids_ranked)}
        rows = [
            {"chunk_id": r.chunk.chunk_id, "dense_rank": dense_rank.get(r.chunk.chunk_id),
             "bm25_rank": bm25_rank.get(r.chunk.chunk_id), "rrf_score": r.rrf_score}
            for r in fused
        ]
        self._data["fusion"] = rows
        self._data["rrf_k"] = rrf_k
        self.mark("fusion", latency_ms)
        if not self.enabled:
            return
        _section("STEP 4 -- RECIPROCAL RANK FUSION (RRF)")
        _kv(
            Candidates=len(fused), **{"RRF k": rrf_k, "Dense Weight": dense_weight, "Sparse Weight": sparse_weight},
            Latency=f"{latency_ms:.1f} ms",
        )
        for row in rows:
            dense_term = f"{dense_weight}*1/({rrf_k}+{row['dense_rank']})" if row["dense_rank"] is not None else None
            bm25_term = f"{sparse_weight}*1/({rrf_k}+{row['bm25_rank']})" if row["bm25_rank"] is not None else None
            terms = [t for t in (dense_term, bm25_term) if t is not None]
            formula = " + ".join(terms) + f" = {row['rrf_score']:.5f}"
            print(f"  chunk={row['chunk_id'][:12]}  dense_rank={row['dense_rank']}  bm25_rank={row['bm25_rank']}  {formula}")

    def log_rerank(self, provider_name: str, fused: list, reranked: list, latency_ms: float, budget_applied: int) -> None:
        selected_ids = {r.chunk.chunk_id for r in reranked}
        dropped = [c.chunk.chunk_id for c in fused if c.chunk.chunk_id not in selected_ids]
        fusion_candidates = len(self._data.get("fusion", []))
        self._data["rerank"] = {
            "provider": provider_name,
            "selected": [{"chunk_id": r.chunk.chunk_id, "score": r.rerank_score, "final_rank": r.final_rank} for r in reranked],
            "dropped": dropped,
            "budget_applied": budget_applied,
            "fusion_candidates": fusion_candidates,
            "sent_to_reranker": len(fused),
            "returned": len(reranked),
        }
        self.mark("rerank", latency_ms)
        if not self.enabled:
            return
        _section("STEP 5 -- CROSS-ENCODER / RERANKER")
        _kv(Provider=provider_name, Candidates=len(fused), Selected=len(reranked), Latency=f"{latency_ms:.1f} ms")
        for r in reranked:
            score = "n/a" if r.rerank_score is None else f"{r.rerank_score:.4f}"
            print(f"  SELECTED  chunk={r.chunk.chunk_id[:12]}  rank={r.final_rank}  score={score}")
        for cid in dropped:
            print(f"  DROPPED   chunk={cid[:12]}  reason=not in reranker top-{len(reranked)}")

        dense_count = len(self._data.get("dense", []))
        bm25_count = len(self._data.get("bm25", []))
        saved = fusion_candidates - len(fused)
        _section("RETRIEVAL BUDGET")
        _kv(**{
            "Dense candidates": dense_count,
            "BM25 candidates": bm25_count,
            "Unique fused candidates": fusion_candidates,
            "Configured budget": budget_applied,
            "Sent to reranker": len(fused),
            "Returned": len(reranked),
            "Saved": f"{saved} reranker evaluations",
        })

    def log_query_decomposition(
        self, is_comparative: bool, subqueries: list[str],
        raw_llm_output: str | None, concepts_retrieved: int,
    ) -> None:
        coverage = concepts_retrieved / len(subqueries) if subqueries else 0.0
        self._data["query_decomposition"] = {
            "comparative": is_comparative, "subqueries": subqueries,
            "raw_llm_output": raw_llm_output,
            "concepts_requested": len(subqueries), "concepts_retrieved": concepts_retrieved,
            "coverage": coverage,
        }
        if not self.enabled:
            return
        _section("STEP 1b -- QUERY DECOMPOSITION")
        _kv(Comparative=is_comparative, **{
            "Concepts requested": len(subqueries),
            "Concepts retrieved": concepts_retrieved,
            "Coverage": f"{coverage * 100:.0f}%",
        })
        for i, q in enumerate(subqueries, 1):
            print(f"  {i}. {q!r}")
        if raw_llm_output is not None:
            print(f"\n  Raw decomposition output: {raw_llm_output!r}")

    def log_pruning(self, before: list, after: list) -> None:
        kept_ids = {r.chunk.chunk_id for r in after}
        dropped = [r.chunk.chunk_id for r in before if r.chunk.chunk_id not in kept_ids]
        self._data["pruning"] = {
            "before": len(before), "after": len(after), "dropped": dropped,
        }
        if not self.enabled or len(dropped) == 0:
            return
        _section("STEP 5b -- DYNAMIC CONTEXT PRUNING")
        _kv(**{"Before": len(before), "After": len(after), "Dropped": len(dropped)})
        for r in before:
            kept = r.chunk.chunk_id in kept_ids
            print(f"  {'KEPT   ' if kept else 'PRUNED '} chunk={r.chunk.chunk_id[:12]}  rerank_score={r.rerank_score}")

    def log_prompt(self, prompt: str) -> None:
        approx_tokens = len(prompt) // 4
        self._data["prompt"] = {"text": prompt, "chars": len(prompt), "approx_tokens": approx_tokens}
        if not self.enabled:
            return
        _section("STEP 6 -- PROMPT BUILDER")
        _kv(Characters=len(prompt), **{"Approx Tokens": approx_tokens})
        print("\n--- PROMPT ---")
        print(prompt)

    def log_generation(self, provider_name: str, model: str, raw_output: str, latency_ms: float) -> None:
        self._data["generation"] = {"provider": provider_name, "model": model, "latency_ms": latency_ms, "raw_output": raw_output}
        self.mark("generation", latency_ms)
        if not self.enabled:
            return
        _section("STEP 7 -- LLM GENERATION")
        latency_label = f"{latency_ms / 1000:.2f} sec" if latency_ms >= 1000 else f"{latency_ms:.1f} ms"
        _kv(Provider=provider_name, Model=model, Latency=latency_label)
        print("\n--- RAW OUTPUT (pre-parse) ---")
        print(raw_output)

    def log_parse(self, success: bool, claims_count: int = 0, quotes_count: int = 0, error: str | None = None) -> None:
        self._data["parse"] = {"success": success, "claims": claims_count, "quotes": quotes_count, "error": error}
        if not self.enabled:
            return
        _section("STEP 8 -- JSON PARSING")
        if success:
            _kv(Status="SUCCESS", Claims=claims_count, **{"Supporting Quotes": quotes_count})
        else:
            _kv(Status="FAILED", Reason=error)

    def log_verification(self, verification) -> None:
        rows = [
            {"text": cr.claim.text, "citation_ids": cr.claim.citation_ids, "doc_ids_valid": cr.doc_ids_valid,
             "quote_match_score": cr.quote_match_score, "passed": cr.passed, "failure_reason": cr.failure_reason}
            for cr in verification.claim_results
        ]
        ratio = (
            verification.verified_claims / verification.total_claims
            if verification.total_claims else 0.0
        )
        self._data["verification"] = {
            "total": verification.total_claims, "verified": verification.verified_claims,
            "failed": verification.failed_claims, "verification_ratio": ratio, "claims": rows,
        }
        if not self.enabled:
            return
        _section("STEP 9 -- CITATION VERIFICATION")
        _kv(**{"Total Claims": verification.total_claims, "Verified": verification.verified_claims, "Failed": verification.failed_claims})
        for i, row in enumerate(rows, 1):
            status = "PASS" if row["passed"] else "FAIL"
            reason = f"  reason={row['failure_reason']}" if row["failure_reason"] else ""
            print(f"\n  Claim {i} [{status}]  citations={row['citation_ids']}  quote_match={row['quote_match_score']:.3f}{reason}")
            print(f"    {row['text'][:160]!r}")
        _section("VERIFICATION SUMMARY")
        _kv(**{
            "Claims generated": verification.total_claims,
            "Claims verified": verification.verified_claims,
            "Claims failed": verification.failed_claims,
            "Verification Ratio": f"{ratio * 100:.0f}%",
        })

    def log_claim_diagnostics(self, rows: list[dict]) -> None:
        self._data["claim_diagnostics"] = rows
        if not self.enabled:
            return
        _section("CLAIM DIAGNOSTICS")
        for row in rows:
            print(f"\nCLAIM {row['claim_index']}")
            _kv(**{
                "citation id": row["citation_id"],
                "chunk id": (row["chunk_id"] or "")[:12] if row["chunk_id"] else None,
                "quote length": row["quote_length"],
                "quote found": row["quote_found"],
                "quote start offset": row["quote_start_offset"],
                "quote end offset": row["quote_end_offset"],
                "crossed boundary": "YES" if row["crossed_boundary"] else "NO",
            })

    def log_confidence(self, confidence) -> None:
        self._data["confidence"] = {
            "retrieval": confidence.retrieval, "citations": confidence.citations,
            "coverage": confidence.coverage, "overall": confidence.overall,
        }
        if not self.enabled:
            return
        _section("STEP 10 -- CONFIDENCE SCORING")
        _kv(
            Retrieval=f"{confidence.retrieval:.2f}", Citations=f"{confidence.citations:.2f}",
            Coverage=f"{confidence.coverage:.2f}", Overall=f"{confidence.overall:.2f}",
        )

    def log_citation_check(self, inline_ids: list[str], structured_ids: list[str], status: str) -> None:
        self._data["citation_check"] = {
            "inline": inline_ids, "structured": structured_ids, "status": status,
        }
        if not self.enabled:
            return
        _section("CITATION STATUS")
        _kv(Status=status, Inline=inline_ids or "[]", Structured=structured_ids or "[]")
        if status != "ok":
            print("Action            : no mutation performed")

    def log_summary(self, answer: str | None, chunks_used: int, documents_used: int) -> None:
        self._data["summary"] = {
            "answer": answer, "chunks_used": chunks_used, "documents_used": documents_used,
        }
        if not self.enabled:
            return
        _section("REQUEST SUMMARY")
        _kv(
            Question=self.question,
            Answer=(answer or "")[:200],
            **{"Chunks Used": chunks_used, "Documents Used": documents_used, "Trace ID": self.request_id},
        )
        print(_EXECUTION_GRAPH)

    def finish(self) -> dict:
        total_ms = (time.perf_counter() - self._t0) * 1000
        self._timings["total"] = total_ms
        self._data["timings_ms"] = self._timings
        self._data["runtime"] = {"python": platform.python_version(), "platform": platform.platform()}
        if self.enabled:
            _section("TIMING")
            for name, ms in self._timings.items():
                print(f"  {name:<16}: {ms:.1f} ms")
            _section("REQUEST END")
        if self.enabled:
            self._save()
        return self._data

    def _save(self) -> None:
        self._traces_dir.mkdir(parents=True, exist_ok=True)
        fname = self.started_at.strftime("%Y-%m-%dT%H-%M-%S") + f"-{self.request_id}.json"
        with open(self._traces_dir / fname, "w") as f:
            json.dump(self._data, f, indent=2, default=str)
