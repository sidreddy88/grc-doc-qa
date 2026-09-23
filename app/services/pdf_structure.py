"""Structure recovery for extracted PDF text: running headers, table of contents, section headings.

Everything here is best-effort metadata. If a document has no recognizable
header or ToC, these functions return empty results and the caller carries on
without section labels.
"""

import re
from collections import Counter
from dataclasses import dataclass

from app.services.text import fold_for_match, fold_quotes, normalize_whitespace

_TOC_MARKER = "table of contents"
_TOC_ENTRY = re.compile(r"^\s*(?P<title>.*?\S)\s*(?:\.{2,}\s*|\s{2,})(?P<page>\d{1,4})\s*$")
_HEADING_LOOKAHEAD = 4


@dataclass(frozen=True, slots=True)
class Heading:
    page: int
    offset: int
    title: str


def find_running_header(page_texts: list[str], min_share: float = 0.6, max_words: int = 40) -> str | None:
    """Return the longest word prefix shared by most pages (e.g. a report title repeated on every page)."""
    word_lists = [text.split()[:max_words] for text in page_texts if text]
    if len(word_lists) < 3:
        return None
    needed = max(3, int(len(word_lists) * min_share))
    for length in range(max_words, 2, -1):
        prefixes = Counter(tuple(words[:length]) for words in word_lists if len(words) >= length)
        if not prefixes:
            continue
        prefix, count = prefixes.most_common(1)[0]
        if count >= needed:
            return " ".join(prefix)
    return None


def strip_page_furniture(text: str, header: str | None, page_number: int) -> str:
    text = normalize_whitespace(text)
    if header and text.startswith(header):
        text = text[len(header) :].lstrip()
    return re.sub(rf"\s*\b{page_number}$", "", text)


def is_toc_start(page_text: str) -> bool:
    return _TOC_MARKER in fold_for_match(normalize_whitespace(page_text))


def parse_toc_lines(layout_text: str) -> list[str]:
    """Parse ToC entry titles from layout-mode text; wrapped titles are joined onto their numbered line."""
    titles: list[str] = []
    pending: list[str] = []
    for raw_line in layout_text.splitlines():
        line = raw_line.strip()
        if not line or _TOC_MARKER in line.lower():
            pending.clear()
            continue
        match = _TOC_ENTRY.match(line)
        if match:
            titles.append(normalize_whitespace(" ".join([*pending, match.group("title")])))
            pending.clear()
        else:
            pending.append(line)
    return titles


def looks_like_toc_page(layout_text: str, min_share: float = 0.6) -> bool:
    lines = [line for line in layout_text.splitlines() if line.strip()]
    if len(lines) < 5:
        return False
    entries = sum(1 for line in lines if _TOC_ENTRY.match(line))
    return entries / len(lines) >= min_share


def locate_headings(pages: list[tuple[int, str]], titles: list[str]) -> list[Heading]:
    """Find where each ToC title actually occurs in the body text.

    ToC page numbers are unreliable (often offset from the physical page), so
    titles are matched against page text in document order instead. Matching is
    case-sensitive because headings reproduce the ToC's casing while incidental
    mentions in prose usually don't ("Opinion" vs "express an opinion"). The next
    expected title wins over later ones; a small lookahead lets matching skip
    titles that never appear in the body (e.g. ones only on divider slides).
    """
    patterns = [re.compile(rf"(?<!\w){re.escape(fold_quotes(title))}(?!\w)") for title in titles]
    headings: list[Heading] = []
    next_title = 0
    for page_number, text in pages:
        folded = fold_quotes(text)
        position = 0
        while next_title < len(titles):
            best: tuple[int, int] | None = None
            for index in range(next_title, min(next_title + _HEADING_LOOKAHEAD, len(titles))):
                match = patterns[index].search(folded, position)
                if match:
                    best = (index, match.start())
                    break
            if best is None:
                break
            index, offset = best
            headings.append(Heading(page=page_number, offset=offset, title=titles[index]))
            position = offset + len(titles[index])
            next_title = index + 1
    return headings


def split_at_headings(
    pages: list[tuple[int, str]], headings: list[Heading]
) -> list[tuple[int, str | None, str]]:
    """Cut page text at heading offsets so every segment belongs to exactly one section."""
    by_page: dict[int, list[Heading]] = {}
    for heading in headings:
        by_page.setdefault(heading.page, []).append(heading)

    segments: list[tuple[int, str | None, str]] = []
    current_section: str | None = None
    for page_number, text in pages:
        cursor = 0
        for heading in by_page.get(page_number, []):
            before = text[cursor : heading.offset].strip()
            if before:
                segments.append((page_number, current_section, before))
            current_section = heading.title
            cursor = heading.offset
        rest = text[cursor:].strip()
        if rest:
            segments.append((page_number, current_section, rest))
    return segments
