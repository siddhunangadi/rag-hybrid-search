import json
import re

# Deliberately broad: false negatives silently fall back to single-query
# retrieval (today's unchanged behavior), so the cost of over-matching a
# borderline question is low, while under-matching a genuine comparison
# reproduces the original bug. Groups: direct comparison words, relational
# phrasing ("related to", "vs"), superiority/ranking ("outperform",
# "better", "worse"), and cost/benefit framing ("pros and cons",
# "advantages", "tradeoffs", "relative performance", "correlation").
_COMPARATIVE_RE = re.compile(
    r"\b("
    r"compare|comparison|comparative|difference|differ|differs|different|"
    r"versus|vs\.?|across|contrast|relationship|related|relative|between|"
    r"outperform|underperform|better|worse|superior|inferior|"
    r"pros and cons|advantages?|disadvantages?|tradeoffs?|trade-offs?|"
    r"correlation|correlate"
    r")\b",
    re.IGNORECASE,
)


def is_comparative_query(question: str) -> bool:
    """Cheap keyword check for questions that need evidence from more than
    one concept/section to answer well (e.g. "compare X and Y across A, B, C",
    "which performs better", "what's the tradeoff", "how is X related to Y").

    False negatives just mean the pipeline falls back to single-query
    retrieval (today's behavior) -- never a hard failure, so a plain
    regex is an acceptable heuristic here rather than an LLM call.
    """
    return bool(_COMPARATIVE_RE.search(question))


_DECOMPOSITION_PROMPT_TEMPLATE = """The following question asks for a comparison across multiple concepts, sections, or sources. List the distinct concepts that need to be retrieved separately to answer it well, as a JSON array of short search-query strings (no more than {max_subqueries} items), ordered from most important to least important. Respond with ONLY the JSON array, no prose.

Question: {question}

Example:
Question: How do function-level and class-level detection patterns differ across RQ1 and RQ2?
["function-level detection patterns", "class-level detection patterns", "RQ1 findings", "RQ2 findings"]
"""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def decompose_query(
    question: str, generation_provider, max_subqueries: int = 4, capture: dict | None = None,
) -> list[str]:
    """Break a comparative question into independent sub-queries so each
    referenced concept gets its own retrieval pass instead of being averaged
    into one dominant embedding.

    Never raises: any failure -- provider exception, malformed JSON, empty
    result, or a single sub-query that just echoes the question back --
    falls back to ``[question]``, identical to today's non-comparative
    single-retrieve behavior. Deliberately does NOT reject sub-queries for
    being short or "generic" by word count -- that conflates length with
    specificity (see Step 7 note above) and would drop useful short
    identifiers (e.g. "RQ1", "SOC2") along with genuinely vague output.

    When `capture` is provided, `capture["raw"]` is set to the provider's
    raw response string (or `None` if `generation_provider.generate` itself
    raised) regardless of whether validation later rejects it -- callers
    that want to debug *why* decomposition fell back can inspect exactly
    what the LLM returned.
    """
    prompt = _DECOMPOSITION_PROMPT_TEMPLATE.format(question=question, max_subqueries=max_subqueries)
    raw = None
    try:
        raw = generation_provider.generate(prompt)
        parsed = json.loads(raw)
    except Exception:
        if capture is not None:
            capture["raw"] = raw
        return [question]

    if capture is not None:
        capture["raw"] = raw

    if not isinstance(parsed, list) or not parsed:
        return [question]

    subqueries = [str(item) for item in parsed if str(item).strip()]
    if not subqueries:
        return [question]

    if len(subqueries) == 1 and _normalize(subqueries[0]) == _normalize(question):
        return [question]

    return subqueries[:max_subqueries]
