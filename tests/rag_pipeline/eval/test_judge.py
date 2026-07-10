import json

from rag_pipeline.eval.judge import judge_answer


class CannedJudgeProvider:
    def __init__(self, response: str):
        self._response = response

    def generate(self, prompt, **kwargs):
        return self._response


def test_judge_answer_parses_valid_verdict():
    response = json.dumps({"verdict": "CORRECT", "reasoning": "Matches the gold answer."})
    provider = CannedJudgeProvider(response)

    result = judge_answer("What is X?", "X is a thing.", "X is a thing.", provider)

    assert result.verdict == "CORRECT"
    assert result.reasoning == "Matches the gold answer."
    assert result.raw_response == response
    assert "What is X?" in result.prompt


def test_judge_answer_accepts_all_valid_verdicts():
    for verdict in ("CORRECT", "PARTIAL", "INCORRECT", "UNSUPPORTED"):
        response = json.dumps({"verdict": verdict, "reasoning": "..."})
        result = judge_answer("Q", "gold", "model", CannedJudgeProvider(response))
        assert result.verdict == verdict


def test_judge_answer_falls_back_to_incorrect_on_malformed_json():
    provider = CannedJudgeProvider("not valid json at all")

    result = judge_answer("What is X?", "X is a thing.", "garbage", provider)

    assert result.verdict == "INCORRECT"
    assert "parse" in result.reasoning.lower()
    assert result.raw_response == "not valid json at all"


def test_judge_answer_falls_back_to_incorrect_on_unrecognized_verdict():
    response = json.dumps({"verdict": "MAYBE", "reasoning": "..."})
    provider = CannedJudgeProvider(response)

    result = judge_answer("Q", "gold", "model", provider)

    assert result.verdict == "INCORRECT"
    assert "MAYBE" in result.reasoning


def test_judge_answer_falls_back_to_incorrect_on_missing_verdict_key():
    response = json.dumps({"reasoning": "no verdict field here"})
    provider = CannedJudgeProvider(response)

    result = judge_answer("Q", "gold", "model", provider)

    assert result.verdict == "INCORRECT"
