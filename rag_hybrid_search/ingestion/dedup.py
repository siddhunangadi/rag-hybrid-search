from difflib import SequenceMatcher

from rag_hybrid_search.models import Chunk


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b).ratio()


def is_duplicate(
    candidate: Chunk,
    candidate_embedding: list[float],
    existing: list[tuple[Chunk, list[float]]],
    cosine_threshold: float,
    text_threshold: float,
) -> bool:
    for existing_chunk, existing_embedding in existing:
        cosine_sim = _cosine(candidate_embedding, existing_embedding)
        if cosine_sim <= cosine_threshold:
            continue
        text_sim = _text_similarity(candidate.text, existing_chunk.text)
        if text_sim > text_threshold:
            return True
    return False
