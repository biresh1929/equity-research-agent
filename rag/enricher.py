"""Groq llama-8b chunk enrichment with hash-based caching."""

import json
import hashlib
import logging
import time
from pathlib import Path

# Groq free tier: 30 req/min. Sleep 2.2s between calls → ~27 req/min, safely under limit.
_RATE_SLEEP = 2.2
# Maximum non-table chunks to enrich per filing. Chunks beyond this get a fast fallback
# description so they're still searchable without hammering the API.
_MAX_ENRICH = 60

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from config.settings import settings
from config.prompts import CHUNK_ENRICHMENT_PROMPT

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/enrichment_cache")


def _cache_path(chunk_id: str) -> Path:
    key = hashlib.md5(chunk_id.encode()).hexdigest()
    return _CACHE_DIR / f"{key}.json"


def _load_cache(chunk_id: str) -> dict | None:
    p = _cache_path(chunk_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _save_cache(chunk_id: str, data: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(chunk_id).write_text(json.dumps(data), encoding="utf-8")


def enrich_chunk(chunk: dict) -> dict:
    """
    Add LLM-generated description and search queries to a chunk.

    Uses llama-3.1-8b-instant (fast/cheap) for enrichment.
    Results are cached by chunk_id so re-runs don't re-call the API.
    """
    chunk_id = chunk["chunk_id"]
    cached = _load_cache(chunk_id)
    if cached:
        chunk.update(cached)
        return chunk

    llm = ChatGroq(
        model=settings.fast_model,
        temperature=0.0,
        api_key=settings.groq_api_key,
    )

    prompt = CHUNK_ENRICHMENT_PROMPT.format(
        company_name=chunk.get("company", ""),
        filing_type=chunk.get("filing_type", ""),
        chunk_text=chunk["text"][:1200],
    )

    try:
        time.sleep(_RATE_SLEEP)  # stay under Groq free-tier 30 req/min
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = json.loads(response.content)
        description = parsed.get("description", "")
        queries = parsed.get("queries", [])
    except Exception as e:
        logger.warning("Enrichment failed for %s: %s", chunk_id, e)
        description = ""
        queries = []

    enrichment = {"description": description, "queries": queries}
    _save_cache(chunk_id, enrichment)
    chunk.update(enrichment)
    return chunk


def _fallback_description(chunk: dict) -> None:
    """Fast no-API fallback for chunks beyond the enrichment cap."""
    section = chunk.get("section", "filing")
    company = chunk.get("company", "")
    chunk["description"] = f"{company} {section} content"
    chunk["queries"] = [
        f"{company} {section}",
        f"{section} financial information",
    ]


def enrich_chunks(chunks: list[dict]) -> list[dict]:
    """
    Enrich chunks with LLM-generated descriptions and search queries.
    Tables get a fast fallback (self-descriptive). Non-table chunks are enriched
    up to _MAX_ENRICH; beyond that they get a section-based fallback to avoid
    hammering the Groq free-tier rate limit (30 req/min).
    """
    enriched = []
    llm_calls = 0

    for chunk in chunks:
        if chunk.get("is_table"):
            chunk["description"] = f"Financial table from {chunk.get('section', 'filing')}"
            chunk["queries"] = [
                f"financial data table {chunk.get('section', '')}",
                f"{chunk.get('company', '')} financial figures",
            ]
        elif llm_calls < _MAX_ENRICH:
            # Check cache first — cached chunks don't count against the cap
            cached = _load_cache(chunk["chunk_id"])
            if cached:
                chunk.update(cached)
            else:
                enrich_chunk(chunk)
                llm_calls += 1
        else:
            _fallback_description(chunk)
        enriched.append(chunk)

    logger.info("Enriched %d chunks via LLM (%d used fallback)", llm_calls, len(enriched) - llm_calls)
    return enriched
