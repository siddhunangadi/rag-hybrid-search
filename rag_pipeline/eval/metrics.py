from rag_pipeline.eval.judge import judge_answer
from rag_pipeline.eval.questions import EvalQuestion
from rag_pipeline.models import RagAnswer

_VERDICT_SCORES = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0, "UNSUPPORTED": 0.0}


def verdict_score(verdict: str) -> float:
    return _VERDICT_SCORES[verdict]


def citation_precision_recall_f1(predicted: list[str], expected: list[str]) -> tuple[float, float, float]:
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    predicted_set, expected_set = set(predicted), set(expected)
    true_positives = len(predicted_set & expected_set)

    precision = true_positives / len(predicted_set) if predicted_set else 0.0
    recall = true_positives / len(expected_set) if expected_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def build_retrieval_record(trace_data: dict) -> dict:
    timings = trace_data.get("timings_ms", {})
    rerank = trace_data.get("rerank", {})
    pruning = trace_data.get("pruning", {})
    prompt = trace_data.get("prompt", {})
    summary = trace_data.get("summary", {})

    return {
        "retrieved_chunk_ids": [r["chunk_id"] for r in trace_data.get("dense", [])],
        "reranked_chunk_ids": [r["chunk_id"] for r in rerank.get("selected", [])],
        "chunks_used": summary.get("chunks_used"),
        "document_ids_used": summary.get("documents_used"),
        "context_size_chars": prompt.get("chars"),
        "pruning": pruning,
        "retrieval_latency_ms": timings.get("dense_search"),
        "rerank_latency_ms": timings.get("rerank"),
        "generation_latency_ms": timings.get("generation"),
    }


def evaluate_question(
    question: EvalQuestion, rag_answer: RagAnswer, trace_data: dict, latency_ms: float,
    judge_provider, judge_prompt_version: str = "v1",
) -> dict:
    if rag_answer.error is not None:
        return error_record(question, error_type="pipeline_error", error_message=rag_answer.error)

    predicted_citations = rag_answer.citations
    expected_citations = question.expected.citation_doc_ids
    precision, recall, f1 = citation_precision_recall_f1(predicted_citations, expected_citations)

    claim_results = rag_answer.verification.claim_results
    verification_pass = bool(claim_results) and all(cr.passed for cr in claim_results)

    judge_result = judge_answer(
        question.question, question.expected.answer, rag_answer.answer or "",
        judge_provider, prompt_version=judge_prompt_version,
    )

    return {
        "id": question.id,
        "question": question.question,
        "category": question.category,
        "expected": {"answer": question.expected.answer, "citation_doc_ids": expected_citations},
        "model_answer": rag_answer.answer,
        "citations": predicted_citations,
        "verification": rag_answer.verification.model_dump(),
        "confidence": rag_answer.confidence.model_dump(),
        "status": "success",
        "error_type": None,
        "error_message": None,
        "objective_metrics": {
            "latency_ms": latency_ms,
            "citation_precision": precision,
            "citation_recall": recall,
            "citation_f1": f1,
            "verification_pass": verification_pass,
            "coverage": rag_answer.confidence.coverage,
        },
        "judge": {
            "verdict": judge_result.verdict,
            "reasoning": judge_result.reasoning,
            "prompt": judge_result.prompt,
            "raw_response": judge_result.raw_response,
        },
        "retrieval": build_retrieval_record(trace_data),
    }


def error_record(question: EvalQuestion, error_type: str, error_message: str) -> dict:
    """Built by the driver (Task 6) when the pipeline call itself raises,
    so one question's failure doesn't abort the whole eval run."""
    return {
        "id": question.id,
        "question": question.question,
        "category": question.category,
        "expected": {"answer": question.expected.answer, "citation_doc_ids": question.expected.citation_doc_ids},
        "model_answer": None,
        "citations": [],
        "verification": None,
        "confidence": None,
        "status": "error",
        "error_type": error_type,
        "error_message": error_message,
        "objective_metrics": None,
        "judge": None,
        "retrieval": None,
    }
