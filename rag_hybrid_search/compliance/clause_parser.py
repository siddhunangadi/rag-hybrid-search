import re

from rag_hybrid_search.compliance.regulation_models import (
    ClauseParseResult,
    ClauseSpan,
    LegalMetadata,
)

_ARTICLE_RE = re.compile(r"^(?:Article|ARTICLE|Art\.)\s+(\d+[A-Za-z]?)\s*$", re.MULTILINE)
_SECTION_RE = re.compile(r"^(?:Section|SECTION|Sec\.)\s+([\d.]+)\s*$", re.MULTILINE)
_CHAPTER_RE = re.compile(r"^(?:Chapter|CHAPTER)\s+([IVXLCDM]+|\d+)\s*$", re.MULTILINE)
_ANNEX_RE = re.compile(r"^(?:Annex|ANNEX|Appendix|APPENDIX)\s+([A-Za-z0-9]+)\s*$", re.MULTILINE)
_CLAUSE_RE = re.compile(r"^\(?(\d+(?:\.\d+)*(?:\([a-z]\))?)\)?[\s.:]", re.MULTILINE)

_HEADING_PATTERNS = [
    ("article", _ARTICLE_RE),
    ("section", _SECTION_RE),
    ("chapter", _CHAPTER_RE),
    ("annex", _ANNEX_RE),
]


def parse_clauses(text: str, document_id: str, document_title: str) -> ClauseParseResult:
    """Split text into clause spans using regex heading detection.

    Splits at top-level Article/Section/Chapter/Annex headings, then
    tags nested numbered sub-clauses (e.g. "1.", "5.2(a)") within each
    top-level span. Falls back to a single whole-document clause with
    confidence 0.0 if no heading is recognized anywhere.
    """
    if not text.strip():
        return ClauseParseResult(clauses=[], confidence=0.0)

    matches: list[tuple[int, str, str]] = []
    for label, pattern in _HEADING_PATTERNS:
        for m in pattern.finditer(text):
            matches.append((m.start(), label, m.group(1)))
    matches.sort(key=lambda m: m[0])

    if not matches:
        span = ClauseSpan(
            text=text.strip(),
            metadata=LegalMetadata(document_id=document_id, document_title=document_title),
        )
        return ClauseParseResult(clauses=[span], confidence=0.0)

    boundaries = [m[0] for m in matches] + [len(text)]
    clauses: list[ClauseSpan] = []
    current_article: str | None = None
    current_section: str | None = None

    for i, (start, label, value) in enumerate(matches):
        end = boundaries[i + 1]
        block = text[start:end].strip()
        if label == "article":
            current_article = value
        elif label == "section":
            current_section = value

        sub_clauses = list(_CLAUSE_RE.finditer(block))
        if not sub_clauses:
            clauses.append(
                ClauseSpan(
                    text=block,
                    metadata=LegalMetadata(
                        document_id=document_id,
                        document_title=document_title,
                        article=current_article,
                        section=current_section,
                    ),
                )
            )
            continue

        sub_boundaries = [sc.start() for sc in sub_clauses] + [len(block)]
        for j, sc in enumerate(sub_clauses):
            sub_text = block[sc.start() : sub_boundaries[j + 1]].strip()
            clause_number = sc.group(1)
            full_clause = (
                f"{current_article}.{clause_number}" if current_article else clause_number
            )
            clauses.append(
                ClauseSpan(
                    text=sub_text,
                    metadata=LegalMetadata(
                        document_id=document_id,
                        document_title=document_title,
                        article=current_article,
                        section=current_section,
                        clause=full_clause,
                    ),
                )
            )

    coverage = sum(len(c.text) for c in clauses) / max(len(text), 1)
    confidence = min(1.0, 0.5 + 0.5 * min(coverage, 1.0)) if clauses else 0.0

    return ClauseParseResult(clauses=clauses, confidence=confidence)
