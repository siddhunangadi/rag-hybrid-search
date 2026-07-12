from difflib import SequenceMatcher

from rag_hybrid_search.models import Chunk
from rag_hybrid_search.similarity import cosine_similarity


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
        cosine_sim = cosine_similarity(candidate_embedding, existing_embedding)
        if cosine_sim <= cosine_threshold:
            continue
        text_sim = _text_similarity(candidate.text, existing_chunk.text)
        if text_sim > text_threshold:
            return True
    return False
