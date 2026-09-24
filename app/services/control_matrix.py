"""Row-level extraction of SOC 2 control test matrices.

A Type 2 report's testing section is a table: ID | criterion | control | test | result.
Plain text extraction flattens it; fixed-size windows then cut rows in half and mix
unrelated controls in one chunk, and the same control repeated under every criterion
it maps to crowds the retrieved context with near-copies.

Two extraction paths, both producing one row per distinct control:
- layout (preferred): word positions from the PDF are assigned to columns, so each cell is
  rebuilt exactly, including rows that continue onto the next page;
- text (fallback, when the page layout doesn't yield the expected columns): rows start at a
  criterion ID, and the criterion's wording is recovered as the most common first sentence of
  its rows. Rows cut by a page break keep their cells interleaved on this path.
Rows with identical control and test text are merged, keeping every criterion ID.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app.models import Segment
from app.services.text import fold_for_match, normalize_whitespace

_CRITERION = re.compile(r"(?<![\w.])(?:CC|PI|A|C|P)\d{1,2}\.\d{1,2}(?=\s+[A-Z])")
_CRITERION_WORD = re.compile(r"(?:CC|PI|A|C|P)\d{1,2}\.\d{1,2}")
_RESULT = re.compile(r"No exceptions noted\.|Not tested\.|Exceptions? noted")
_SENTENCE_END = re.compile(r"\.(?=\s+[A-Z])")
_MIN_MARKERS = 2
_MAX_CRITERION_WORDS = 80

# Layout path. Columns: ID, criterion, control, test, result.
_EXPECTED_COLUMNS = 5
_CELL_GAP = 60  # points between consecutive word starts on a line that mark a new cell
_MIN_COLUMN_SHARE = 0.05  # a column start must begin at least this share of the most common start's lines
_COLUMN_SNAP = 3
_ROW_TOP_TOLERANCE = 2


@dataclass(slots=True)
class MatrixRow:
    criteria: list[str]
    body: str
    page: int | None
    criterion_texts: list[str] = field(default_factory=list)

    @property
    def section(self) -> str:
        return f"Test Results: {', '.join(self.criteria)}"

    @property
    def index_context(self) -> str:
        return "\n".join(self.criterion_texts)


def is_control_matrix(text: str) -> bool:
    return len(_CRITERION.findall(text)) >= _MIN_MARKERS and len(_RESULT.findall(text)) >= _MIN_MARKERS


def partition_matrix(segments: list[Segment]) -> tuple[list[Segment], list[Segment]]:
    """Split segments into (prose, matrix). Prose before the first row of a matrix run stays prose."""
    prose: list[Segment] = []
    matrix: list[Segment] = []
    in_matrix = False
    for segment in segments:
        # Once inside the matrix, a page with a single long row (one ID, or only a continuation) still belongs to it.
        continues = in_matrix and bool(_CRITERION.search(segment.text) or _RESULT.search(segment.text))
        if not (is_control_matrix(segment.text) or continues):
            in_matrix = False
            prose.append(segment)
            continue
        first = _CRITERION.search(segment.text)
        if not in_matrix and first and segment.text[: first.start()].strip():
            prose.append(Segment(text=segment.text[: first.start()].strip(), page=segment.page, section=segment.section))
            segment = Segment(text=segment.text[first.start() :], page=segment.page, section=segment.section)
        matrix.append(segment)
        in_matrix = True
    return prose, matrix


# --- text path ---------------------------------------------------------------------------------------------


def rows_from_text(matrix_segments: list[Segment]) -> list[MatrixRow]:
    raw_rows: list[tuple[str, str, int | None]] = []  # (criterion id, text after the id, page)
    for segment in matrix_segments:
        starts = list(_CRITERION.finditer(segment.text))
        lead = (segment.text[: starts[0].start()] if starts else segment.text).strip()
        if lead and raw_rows:  # spill-over of the previous page's last row
            criterion, text, page = raw_rows[-1]
            raw_rows[-1] = (criterion, f"{text} {lead}", page)
        for match, following in zip(starts, [*starts[1:], None], strict=True):
            end = following.start() if following else len(segment.text)
            raw_rows.append((match.group(), segment.text[match.end() : end].strip(), segment.page))

    by_criterion: dict[str, list[str]] = defaultdict(list)
    for criterion, text, _ in raw_rows:
        by_criterion[criterion].append(text)
    wording = {criterion: _criterion_wording(texts) for criterion, texts in by_criterion.items()}

    rows = []
    for criterion, text, page in raw_rows:
        prefix = wording[criterion]
        if prefix and text.startswith(prefix):
            rows.append(MatrixRow([criterion], text[len(prefix) :].strip(), page, [f"{criterion} {prefix}"]))
        else:  # wording unknown (single row) or row cut at a page break: keep it in the body
            rows.append(MatrixRow([criterion], text, page, [criterion]))
    return merge_duplicate_rows([row for row in rows if row.body])


def _criterion_wording(texts: list[str]) -> str:
    """The criterion's own sentence: the most common first sentence across its rows.

    Trust Services Criteria are single sentences, repeated verbatim at the start of every row that
    maps to them. A row cut at a page break has a garbled first sentence, which the majority outvotes.
    """
    firsts = Counter(sentence for text in texts if (sentence := _first_sentence(text)))
    if not firsts:
        return ""
    sentence, count = firsts.most_common(1)[0]
    return sentence if count >= min(2, len(texts)) else ""


def _first_sentence(text: str) -> str:
    match = _SENTENCE_END.search(text)
    if match is None or len(text[: match.end()].split()) > _MAX_CRITERION_WORDS:
        return ""
    return text[: match.end()].strip()


# --- layout path -------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Word:
    x: float
    y: float  # PDF coordinates: larger is higher on the page
    text: str


def rows_from_layout(pages: list[tuple[int, list[Word]]]) -> list[MatrixRow]:
    """Rebuild rows from positioned words of the matrix pages. Returns [] if the layout isn't recognized."""
    furniture = _furniture_lines(pages)
    pages = [(number, _content_words(words, furniture)) for number, words in pages]
    columns = _column_starts([words for _, words in pages])
    if len(columns) != _EXPECTED_COLUMNS:
        return []

    def column_of(word: Word) -> int:
        return max((i for i, start in enumerate(columns) if word.x >= start - _COLUMN_SNAP), default=0)

    rows: list[dict] = []  # {"id", "page", "cells": {column: [(page_seq, -y, x, text)]}}
    for sequence, (number, words) in enumerate(pages):
        ids = sorted((w for w in words if column_of(w) == 0 and _CRITERION_WORD.fullmatch(w.text)), key=lambda w: -w.y)
        tops = [w.y for w in ids]
        page_rows = [{"id": w.text, "page": number, "cells": defaultdict(list)} for w in ids]
        for word in words:
            column = column_of(word)
            if column == 0:
                continue
            owner = next((row for row, top in reversed(list(zip(page_rows, tops))) if top >= word.y - _ROW_TOP_TOLERANCE),
                         None)
            if owner is None:  # above the page's first row: continuation of the previous page's last row
                if not rows:
                    continue  # the table's header row
                owner = rows[-1]
            owner["cells"][column].append((sequence, -word.y, word.x, word.text))
        rows.extend(page_rows)

    def cell(row: dict, column: int) -> str:
        return normalize_whitespace(" ".join(text for *_, text in sorted(row["cells"][column])))

    built = []
    for row in rows:
        body = " ".join(part for part in (cell(row, 2), cell(row, 3), cell(row, 4)) if part)
        if body:
            criterion = cell(row, 1)
            built.append(MatrixRow([row["id"]], body, row["page"], [f"{row['id']} {criterion}".strip()]))
    return merge_duplicate_rows(built)


def _furniture_lines(pages: list[tuple[int, list[Word]]]) -> set[tuple[int, str]]:
    """Running headers/footers: lines with the same position and text on most matrix pages."""
    counts = Counter(key for _, words in pages for key in set(_line_keys(words).values()))
    return {key for key, count in counts.items() if count >= max(3, len(pages) // 2)}


def _content_words(words: list[Word], furniture: set[tuple[int, str]]) -> list[Word]:
    """Drop header/footer lines and bare page numbers."""
    keys = _line_keys(words)
    return [w for w in words if keys[round(w.y)] not in furniture and not keys[round(w.y)][1].isdigit()]


def _line_keys(words: list[Word]) -> dict[int, tuple[int, str]]:
    lines: dict[int, list[Word]] = defaultdict(list)
    for word in words:
        lines[round(word.y)].append(word)
    return {y: (y, " ".join(w.text for w in sorted(line, key=lambda w: w.x))) for y, line in lines.items()}


def _column_starts(pages: list[list[Word]]) -> list[float]:
    """Left edges of the table columns: x positions where a cell's text begins, frequent across pages."""
    starts: Counter[int] = Counter()
    for words in pages:
        lines: dict[int, list[float]] = defaultdict(list)
        for word in words:
            lines[round(word.y)].append(word.x)
        for xs in lines.values():
            xs.sort()
            for previous, x in zip([None, *xs], xs, strict=False):
                if previous is None or x - previous > _CELL_GAP:
                    starts[round(x)] += 1
    if not starts:
        return []
    threshold = max(starts.values()) * _MIN_COLUMN_SHARE
    edges: list[float] = []
    for x in sorted(x for x, count in starts.items() if count >= threshold):
        if not edges or x - edges[-1] > _COLUMN_SNAP:
            edges.append(x)
    return edges


# --- shared --------------------------------------------------------------------------------------------------


def merge_duplicate_rows(rows: list[MatrixRow]) -> list[MatrixRow]:
    merged: dict[str, MatrixRow] = {}
    for row in rows:
        key = fold_for_match(normalize_whitespace(row.body))
        existing = merged.get(key)
        if existing is None:
            merged[key] = row
            continue
        for criterion, text in zip(row.criteria, row.criterion_texts, strict=True):
            if criterion not in existing.criteria:
                existing.criteria.append(criterion)
                existing.criterion_texts.append(text)
    return list(merged.values())
