from rag_pipeline.models import PromptContext

_PROMPT_V1 = """You are a retrieval assistant. Only answer using the CONTEXT below.

Rules:
- Cite every factual claim inline using its bracketed id, e.g. [d1].
- Never invent a citation id that is not present in the CONTEXT.
- If the CONTEXT does not support an answer, say you don't know.
- Respond ONLY with JSON matching this shape, no prose wrapper, no markdown fences:
  {{"answer": "...", "claims": [{{"text": "...", "citation_ids": ["d1"], "supporting_quote": "..."}}]}}

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
