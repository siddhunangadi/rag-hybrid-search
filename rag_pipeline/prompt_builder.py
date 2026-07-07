from rag_pipeline.models import PromptContext

_PROMPT_V1 = """You are a retrieval assistant. Only answer using the CONTEXT below.

Rules:
- Cite every factual claim inline using its bracketed id, e.g. [d1].
- Never invent a citation id that is not present in the CONTEXT.
- If the CONTEXT does not support an answer, say you don't know.
- "supporting_quote" MUST be copied verbatim, character-for-character, from
  the CONTEXT -- same words, same punctuation, same capitalization. Do NOT
  paraphrase or summarize it. If no exact quote in the CONTEXT supports a
  claim, drop that claim (or answer "I don't know" if no claim can be made).
- Respond ONLY with JSON matching this shape, no prose wrapper, no markdown fences:
  {{"answer": "...", "claims": [{{"text": "...", "citation_ids": ["d1"], "supporting_quote": "..."}}]}}

Example:
CONTEXT:
[d1]
Personal information shall be retained no longer than necessary for the
purposes for which it was collected.

QUESTION:
How long can personal information be retained?

CORRECT (supporting_quote copied verbatim):
{{"answer": "Personal information may only be retained as long as necessary for its original purpose [d1].", "claims": [{{"text": "Personal information may only be retained as long as necessary for its original purpose.", "citation_ids": ["d1"], "supporting_quote": "Personal information shall be retained no longer than necessary for the purposes for which it was collected."}}]}}

WRONG (supporting_quote paraphrased instead of copied):
{{"answer": "Personal information may only be retained as long as necessary for its original purpose [d1].", "claims": [{{"text": "Personal information may only be retained as long as necessary for its original purpose.", "citation_ids": ["d1"], "supporting_quote": "Data should be deleted once it's no longer needed for its purpose."}}]}}

CONTEXT:
{context}

QUESTION:
{question}"""

_PROMPT_TEMPLATES = {"v1": _PROMPT_V1}


def build_prompt(
    question: str, context: PromptContext, prompt_version: str = "v1"
) -> str:
    if prompt_version not in _PROMPT_TEMPLATES:
        raise ValueError(f"unknown prompt_version: {prompt_version}")
    template = _PROMPT_TEMPLATES[prompt_version]
    return template.format(context=context.text, question=question)
