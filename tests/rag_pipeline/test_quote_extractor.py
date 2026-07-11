from rag_pipeline.models import Claim, PromptContext
from rag_pipeline.quote_extractor import extract_supporting_quotes


def test_normal_quote_extracted_verbatim_from_single_chunk():
    context = PromptContext(
        text="[d1]\nEmployees get 20 days of paid annual leave per year.",
        doc_id_map={"d1": "chunk-1"},
    )
    claim = Claim(text="Employees get 20 days leave", citation_ids=["d1"])
    [fixed] = extract_supporting_quotes([claim], context)
    assert fixed.supporting_quote == "Employees get 20 days of paid annual leave per year."
    assert fixed.citation_ids == ["d1"]


def test_paraphrased_claim_text_still_extracts_the_supporting_sentence():
    """claim.text can be a paraphrase (the model is explicitly allowed to
    summarize in "answer"/"text"); the extractor should still pick the
    chunk sentence with the strongest lexical overlap as the quote."""
    context = PromptContext(
        text=(
            "[d1]\nOur objective in this work is to bridge these gaps through a "
            "systematic comparative study of LLM-generated code detection. "
            "This is unrelated filler about something else entirely."
        ),
        doc_id_map={"d1": "chunk-1"},
    )
    claim = Claim(
        text="The paper aims to systematically compare methods for detecting LLM-generated code",
        citation_ids=["d1"],
    )
    [fixed] = extract_supporting_quotes([claim], context)
    assert "systematic comparative study of LLM-generated code detection" in fixed.supporting_quote
    assert "unrelated filler" not in fixed.supporting_quote


def test_multiple_claims_each_extract_from_their_own_cited_chunk():
    context = PromptContext(
        text=(
            "[d1]\nPersonal information shall be retained no longer than necessary.\n\n"
            "[d2]\nData subjects may request erasure at any time."
        ),
        doc_id_map={"d1": "chunk-1", "d2": "chunk-2"},
    )
    claims = [
        Claim(text="Retention is time-limited", citation_ids=["d1"]),
        Claim(text="Erasure can be requested", citation_ids=["d2"]),
    ]
    fixed = extract_supporting_quotes(claims, context)
    assert fixed[0].supporting_quote == "Personal information shall be retained no longer than necessary."
    assert fixed[1].supporting_quote == "Data subjects may request erasure at any time."


def test_claim_with_multiple_citation_ids_is_truncated_to_first():
    """A claim citing more than one id has its citation_ids narrowed to the
    first, and the quote is extracted from that one chunk only -- never a
    quote spanning both cited chunks."""
    context = PromptContext(
        text=(
            "[d1]\nThe most critical insight is that granularity dominates.\n\n"
            "[d2]\nWhile RQ2 establishes that structural detection is effective."
        ),
        doc_id_map={"d1": "chunk-1", "d2": "chunk-2"},
    )
    claim = Claim(
        text="Granularity dominates and RQ2 establishes detection is effective",
        citation_ids=["d1", "d2"],
    )
    [fixed] = extract_supporting_quotes([claim], context)
    assert fixed.citation_ids == ["d1"]
    assert "granularity dominates" in fixed.supporting_quote.lower()
    assert "RQ2" not in fixed.supporting_quote


def test_extracted_quote_never_contains_text_from_a_different_chunk():
    """Structural guarantee: even when the neighboring chunk's sentence is a
    much better lexical match for claim.text, the extractor only ever looks
    inside the one chunk the claim actually cites."""
    context = PromptContext(
        text=(
            "[d1]\nThe cafeteria serves lunch from 12pm to 2pm daily.\n\n"
            "[d2]\nEmployees get 20 days of paid annual leave per year."
        ),
        doc_id_map={"d1": "chunk-1", "d2": "chunk-2"},
    )
    claim = Claim(text="Employees get 20 days of paid annual leave", citation_ids=["d1"])
    [fixed] = extract_supporting_quotes([claim], context)
    assert "leave" not in fixed.supporting_quote
    assert fixed.supporting_quote == "The cafeteria serves lunch from 12pm to 2pm daily."


def test_claim_with_unknown_citation_id_gets_empty_quote_not_a_crash():
    context = PromptContext(text="[d1]\nSome fact.", doc_id_map={"d1": "chunk-1"})
    claim = Claim(text="A hallucinated claim", citation_ids=["d99"])
    [fixed] = extract_supporting_quotes([claim], context)
    assert fixed.supporting_quote == ""
    assert fixed.citation_ids == ["d99"]


def test_claim_with_no_citation_ids_gets_empty_quote_not_a_crash():
    context = PromptContext(text="[d1]\nSome fact.", doc_id_map={"d1": "chunk-1"})
    claim = Claim(text="An uncited claim", citation_ids=[])
    [fixed] = extract_supporting_quotes([claim], context)
    assert fixed.supporting_quote == ""
    assert fixed.citation_ids == []
