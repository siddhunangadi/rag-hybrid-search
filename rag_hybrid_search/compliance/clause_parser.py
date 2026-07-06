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
_LETTER_CLAUSE_RE = re.compile(r"^\(([a-z])\)[\s.:]", re.MULTILINE)

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
        return ClauseParseResult(clauses=[span], confidence=0.0, fallback_used=True)

    boundaries = [m[0] for m in matches] + [len(text)]
    clauses: list[ClauseSpan] = []
    current_article: str | None = None
    current_section: str | None = None

    for i, (start, label, value) in enumerate(matches):
        end = boundaries[i + 1]
        block = text[start:end].strip()
        if label == "article":
            current_article = value
            current_section = None
        elif label == "section":
            current_section = value
            current_article = None

        numbered = [(sc.start(), "num", sc.group(1)) for sc in _CLAUSE_RE.finditer(block)]
        lettered = [(sc.start(), "letter", sc.group(1)) for sc in _LETTER_CLAUSE_RE.finditer(block)]
        sub_clauses = sorted(numbered + lettered, key=lambda t: t[0])
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

        sub_boundaries = [sc[0] for sc in sub_clauses] + [len(block)]
        last_number: str | None = None
        for j, (sc_start, kind, sc_value) in enumerate(sub_clauses):
            sub_text = block[sc_start : sub_boundaries[j + 1]].strip()
            if kind == "num":
                last_number = sc_value
                full_clause = (
                    f"{current_article}.{sc_value}" if current_article else sc_value
                )
            else:
                letter = sc_value
                if last_number is not None:
                    full_clause = (
                        f"{current_article}.{last_number}({letter})"
                        if current_article
                        else f"{last_number}({letter})"
                    )
                else:
                    full_clause = f"({letter})"
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
