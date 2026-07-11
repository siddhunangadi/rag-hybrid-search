import json

from rag_pipeline.generation_provider import MockProvider
from rag_pipeline.query_decomposer import decompose_query, is_comparative_query


def test_is_comparative_query_detects_keywords():
    assert is_comparative_query("How do function-level and class-level patterns differ across RQ1, RQ2, RQ3?") is True
    assert is_comparative_query("Compare the results of RQ1 and RQ2.") is True
    assert is_comparative_query("What is the relationship between granularity and detectability?") is True


def test_is_comparative_query_detects_extended_phrasings():
    assert is_comparative_query("How is precision related to recall?") is True
    assert is_comparative_query("Explain function-level vs class-level detection.") is True
    assert is_comparative_query("Which model performs better on this benchmark?") is True
    assert is_comparative_query("Why does GPT-4 outperform GPT-3.5 here?") is True
    assert is_comparative_query("What are the pros and cons of each approach?") is True
    assert is_comparative_query("List the advantages of dense retrieval.") is True
    assert is_comparative_query("What are the tradeoffs of a larger chunk size?") is True
    assert is_comparative_query("What is the relative performance of the two rerankers?") is True
    assert is_comparative_query("Is there a correlation between chunk size and accuracy?") is True
    assert is_comparative_query("Contrast the two retrieval strategies.") is True


def test_is_comparative_query_false_for_simple_factual_question():
    assert is_comparative_query("What does the paper say about chunk overlap?") is False
    assert is_comparative_query("How many days of paid leave do employees get?") is False


def test_decompose_query_parses_llm_json_array():
    canned = json.dumps(["function-level detection patterns", "class-level detection patterns", "RQ1 findings", "RQ2 findings"])
    provider = MockProvider(canned_json=canned)

    subqueries = decompose_query(
        "How do function-level and class-level detection patterns differ across RQ1 and RQ2?",
        provider,
    )

    assert subqueries == [
        "function-level detection patterns",
        "class-level detection patterns",
        "RQ1 findings",
        "RQ2 findings",
    ]


def test_decompose_query_caps_at_max_subqueries():
    canned = json.dumps(["a", "b", "c", "d", "e", "f"])
    provider = MockProvider(canned_json=canned)

    subqueries = decompose_query("compare a, b, c, d, e, f", provider, max_subqueries=3)

    assert subqueries == ["a", "b", "c"]


def test_decompose_query_falls_back_to_original_question_on_malformed_json():
    provider = MockProvider(canned_json="not json at all")

    subqueries = decompose_query("compare X and Y", provider)

    assert subqueries == ["compare X and Y"]


def test_decompose_query_falls_back_to_original_question_on_empty_array():
    provider = MockProvider(canned_json="[]")

    subqueries = decompose_query("compare X and Y", provider)

    assert subqueries == ["compare X and Y"]


def test_decompose_query_falls_back_to_original_question_on_provider_exception():
    class RaisingProvider:
        def generate(self, prompt, **kwargs):
            raise RuntimeError("provider down")

    subqueries = decompose_query("compare X and Y", RaisingProvider())

    assert subqueries == ["compare X and Y"]


def test_decompose_query_rejects_single_subquery_that_echoes_the_question():
    """The LLM sometimes 'decomposes' a question into itself, verbatim or
    with only whitespace/case differences -- that's not a decomposition,
    it's a no-op wearing a JSON array. Treat it as a failed decomposition
    and fall back, same as malformed JSON."""
    question = "How do RQ1 and RQ2 differ?"
    provider = MockProvider(canned_json=json.dumps([question]))

    subqueries = decompose_query(question, provider)

    assert subqueries == [question]


def test_decompose_query_rejects_single_subquery_echo_case_and_whitespace_insensitive():
    question = "How do RQ1 and RQ2 differ?"
    provider = MockProvider(canned_json=json.dumps([f"  {question.upper()}  "]))

    subqueries = decompose_query(question, provider)

    assert subqueries == [question]


def test_decompose_query_trusts_short_but_specific_subqueries():
    """Word count is not a specificity signal: "RQ1 findings", "SOC2
    evidence", and "OAuth flow" are all short (2 words) but carry real
    retrieval signal (proper nouns / alphanumeric identifiers). A
    word-count floor would wrongly reject these -- validation must not
    second-guess sub-query quality beyond the echo check."""
    question = "Compare RQ1, SOC2, and OAuth handling."
    provider = MockProvider(canned_json=json.dumps(["RQ1", "SOC2", "OAuth"]))

    subqueries = decompose_query(question, provider)

    assert subqueries == ["RQ1", "SOC2", "OAuth"]


def test_decompose_query_captures_raw_llm_output_when_requested():
    """The `capture` dict is a debug-mode side channel: even when
    validation rejects the LLM's output and falls back, the caller (trace
    logging) can still see exactly what the LLM returned."""
    canned = json.dumps(["RQ1 findings", "RQ2 findings"])
    provider = MockProvider(canned_json=canned)
    capture: dict = {}

    decompose_query("compare RQ1 and RQ2", provider, capture=capture)

    assert capture["raw"] == canned


def test_decompose_query_captures_raw_output_even_on_validation_failure():
    question = "compare X and Y across many concepts"
    canned = json.dumps([question])
    provider = MockProvider(canned_json=canned)
    capture: dict = {}

    subqueries = decompose_query(question, provider, capture=capture)

    assert subqueries == [question]
    assert capture["raw"] == canned


def test_decompose_query_captures_none_on_provider_exception():
    class RaisingProvider:
        def generate(self, prompt, **kwargs):
            raise RuntimeError("provider down")

    capture: dict = {}
    decompose_query("compare X and Y", RaisingProvider(), capture=capture)

    assert capture["raw"] is None
