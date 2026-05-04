"""Table-aware HTML chunker for SEC filings."""

import re
from pathlib import Path
from bs4 import BeautifulSoup, Tag

from config.settings import settings

# SEC section header patterns (Item 1, Item 1A, etc.)
_SECTION_RE = re.compile(
    r"(?:^|\n)(item\s+\d+[a-z]?\.?\s+[^\n]{3,80})",
    re.IGNORECASE | re.MULTILINE,
)


def _table_to_markdown(table: Tag) -> str:
    """Convert a BeautifulSoup <table> element to pipe-delimited markdown."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"])]
        if cells:
            rows.append("| " + " | ".join(cells) + " |")

    if not rows:
        return ""

    # Insert separator after first row (header)
    if len(rows) > 1:
        sep = "| " + " | ".join(["---"] * rows[0].count("|")) + " |"
        rows.insert(1, sep)

    return "\n".join(rows)


def _sliding_window(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def parse_filing(
    filepath: str,
    company: str,
    ticker: str,
    filing_type: str,
    year: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """
    Parse an SEC filing HTML file into chunks.

    Tables are extracted as atomic markdown chunks (not split).
    Text sections use sliding-window chunking within each Item section.

    Returns list of chunk dicts with metadata.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    content = Path(filepath).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(content, "html.parser")

    chunks: list[dict] = []
    chunk_id = 0
    doc_id = f"{ticker}_{filing_type}_{year}"

    def _make_chunk(text: str, section: str, is_table: bool) -> dict:
        nonlocal chunk_id
        c = {
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_{chunk_id:04d}",
            "text": text,
            "section": section,
            "is_table": is_table,
            "company": company,
            "ticker": ticker,
            "filing_type": filing_type,
            "year": year,
        }
        chunk_id += 1
        return c

    # Track current section name
    current_section = "preamble"

    # Walk the document top-level elements to preserve order
    body = soup.find("body") or soup
    for element in body.descendants:
        if not isinstance(element, Tag):
            continue

        # Detect section headers
        if element.name in ("h1", "h2", "h3", "h4", "p", "div"):
            text = element.get_text(" ", strip=True)
            if _SECTION_RE.match(text):
                current_section = text[:80].strip()

        # Process tables as atomic chunks
        if element.name == "table":
            md = _table_to_markdown(element)
            if len(md) > 50:  # skip empty/tiny tables
                chunks.append(_make_chunk(md, current_section, is_table=True))
            # Mark table as processed so descendants are skipped
            for desc in element.find_all(True):
                desc.decompose()

    # Second pass: extract all remaining text (non-table) by section
    # Re-parse since we decomposed tables
    soup2 = BeautifulSoup(content, "html.parser")
    for table in soup2.find_all("table"):
        table.replace_with(soup2.new_tag("span"))  # blank out tables

    plain_text = soup2.get_text("\n", strip=True)

    # Split by section headers
    parts = _SECTION_RE.split(plain_text)
    section_name = "preamble"
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        if _SECTION_RE.match(stripped):
            section_name = stripped[:80]
        else:
            for window in _sliding_window(stripped, chunk_size, chunk_overlap):
                if len(window) > 100:  # skip tiny fragments
                    chunks.append(_make_chunk(window, section_name, is_table=False))

    return chunks
