import json
from dataclasses import dataclass

from rag_pipeline.generation_provider import GenerationProvider

_VALID_VERDICTS = {"CORRECT", "PARTIAL", "INCORRECT", "UNSUPPORTED"}

_JUDGE_PROMPT_TEMPLATE = """You are grading a RAG system's answer against a gold reference answer.

Question: {question}

Gold reference answer: {expected_answer}

System's answer: {model_answer}

Score the system's answer as one of:
- CORRECT: matches the gold answer's meaning (paraphrasing is fine).
- PARTIAL: partially correct, missing some aspect of the gold answer.
- INCORRECT: wrong, contradicts the gold answer.
- UNSUPPORTED: the system's answer contains a claim not grounded in the
  gold reference or the question (e.g. a hallucinated fact), regardless
  of whether it happens to be plausible-sounding.

Respond with ONLY a JSON object: {{"verdict": "<one of the four above>", "reasoning": "<one sentence>"}}
"""


@dataclass
class JudgeVerdict:
    verdict: str
    reasoning: str
    prompt: str
    raw_response: str


def judge_answer(
    question: str, expected_answer: str, model_answer: str,
    judge_provider: GenerationProvider, prompt_version: str = "v1",
) -> JudgeVerdict:
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=question, expected_answer=expected_answer, model_answer=model_answer,
    )
    raw_response = judge_provider.generate(prompt)

    try:
        parsed = json.loads(raw_response)
        verdict = parsed["verdict"]
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, TypeError):
        return JudgeVerdict(
            verdict="INCORRECT", reasoning="Judge response failed to parse as JSON.",
            prompt=prompt, raw_response=raw_response,
        )

    if verdict not in _VALID_VERDICTS:
        return JudgeVerdict(
            verdict="INCORRECT",
            reasoning=f"Judge returned an unrecognized verdict {verdict!r}.",
            prompt=prompt, raw_response=raw_response,
        )

    return JudgeVerdict(verdict=verdict, reasoning=reasoning, prompt=prompt, raw_response=raw_response)
