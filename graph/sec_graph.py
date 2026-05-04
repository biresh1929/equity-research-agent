"""SEC filing analysis LangGraph — Article 2 fan-out RAG pipeline."""

import logging
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from config.settings import settings
from graph.state import SECState
from rag.edgar_downloader import EDGARDownloader
from rag.document_parser import parse_filing
from rag.enricher import enrich_chunks
from rag.chroma_store import ChromaStore, safe_collection_name
from rag.bm25_store import get_or_build
from rag.sec_experts import (
    risk_analyst_node,
    sentiment_analyst_node,
    fundamental_analyst_node,
    math_agent_node,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node: resolve CIK
# ---------------------------------------------------------------------------

def resolve_cik_node(state: SECState) -> dict:
    downloader = EDGARDownloader()
    cik = downloader.get_cik_for_ticker(state["ticker"])
    return {"cik": cik}


# ---------------------------------------------------------------------------
# Node: download filing
# ---------------------------------------------------------------------------

def download_filing_node(state: SECState) -> dict:
    downloader = EDGARDownloader()
    filing_type = state.get("filing_type", "10-K")
    try:
        cik, company_name, accession, local_path = downloader.download_filing(
            state["ticker"], filing_type
        )
        return {
            "cik": cik,
            "company_name": company_name,
            "filing_path": local_path,
            "collection_name": safe_collection_name(state["ticker"], filing_type, cik),
        }
    except Exception as e:
        logger.error("Filing download failed: %s", e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Node: chunk filing
# ---------------------------------------------------------------------------

def chunk_filing_node(state: SECState) -> dict:
    if state.get("error"):
        return {}

    collection_name = state["collection_name"]

    # If ChromaDB collection already has data, skip the entire parse→enrich→index pipeline.
    # The BM25 index will be rebuilt from ChromaDB on first expert query this session.
    existing_store = ChromaStore(collection_name)
    existing_count = existing_store.count()
    if existing_count > 0:
        logger.info(
            "Collection %s already has %d docs — skipping re-index",
            collection_name, existing_count,
        )
        _INDEXED_COLLECTIONS.add(collection_name)
        return {}

    year = str(datetime.now().year)
    chunks = parse_filing(
        filepath=state["filing_path"],
        company=state.get("company_name", state["ticker"]),
        ticker=state["ticker"],
        filing_type=state.get("filing_type", "10-K"),
        year=year,
    )
    _CHUNK_REGISTRY[collection_name] = chunks
    logger.info("Parsed %d chunks from %s", len(chunks), state["filing_path"])
    return {}


# ---------------------------------------------------------------------------
# Node: enrich chunks
# ---------------------------------------------------------------------------

_CHUNK_REGISTRY: dict[str, list[dict]] = {}
# Collections that already have indexed data — skip re-chunk/enrich/index for these
_INDEXED_COLLECTIONS: set[str] = set()


def enrich_chunks_node(state: SECState) -> dict:
    if state.get("error"):
        return {}

    collection_name = state["collection_name"]

    if collection_name in _INDEXED_COLLECTIONS:
        logger.info("Collection %s already indexed — skipping enrichment", collection_name)
        return {}

    chunks = _CHUNK_REGISTRY.get(collection_name, [])
    if not chunks:
        return {"error": "No chunks found — parsing may have failed"}

    enriched = enrich_chunks(chunks)
    _CHUNK_REGISTRY[collection_name] = enriched
    return {}


# ---------------------------------------------------------------------------
# Node: build index
# ---------------------------------------------------------------------------

def build_index_node(state: SECState) -> dict:
    if state.get("error"):
        return {}

    collection_name = state["collection_name"]

    if collection_name in _INDEXED_COLLECTIONS:
        logger.info("Collection %s already indexed — rebuilding BM25 from ChromaDB", collection_name)
        from rag.bm25_store import _REGISTRY as _BM25_REG
        if collection_name not in _BM25_REG:
            store = ChromaStore(collection_name)
            chunks = store.get_all_chunks()
            get_or_build(collection_name, chunks)
        return {}

    chunks = _CHUNK_REGISTRY.get(collection_name, [])
    if not chunks:
        return {"error": "No enriched chunks to index"}

    # ChromaDB upsert
    store = ChromaStore(collection_name)
    store.upsert_chunks(chunks)

    # BM25 registry (loaded into memory for this session)
    get_or_build(collection_name, chunks)

    _INDEXED_COLLECTIONS.add(collection_name)
    logger.info("Indexed %d chunks for %s", len(chunks), collection_name)
    return {}


# ---------------------------------------------------------------------------
# Fan-out to expert analysts
# ---------------------------------------------------------------------------

def fan_out_experts(state: SECState) -> list[Send]:
    if state.get("error"):
        return []
    return [
        Send("risk_analyst", state),
        Send("sentiment_analyst", state),
        Send("fundamental_analyst", state),
        Send("math_agent", state),
    ]


# ---------------------------------------------------------------------------
# Node: aggregate and generate report
# ---------------------------------------------------------------------------

_FILING_REPORT_SYSTEM = """You are a senior equity research director synthesising expert analyses of an SEC filing.
You have four specialist reports: Risk, Sentiment, Fundamentals, and Mathematical computations.
Write a comprehensive 2-page investment-grade filing summary.

Structure:
1. Executive Summary (3 bullets)
2. Key Risk Factors (top 5 from risk analyst)
3. Management Sentiment (top signals from sentiment analyst)
4. Financial Performance (key metrics from fundamental analyst + math computations)
5. Conclusion: is the filing broadly positive, neutral, or concerning for investors?

Use specific numbers. Be direct. No fluff."""

_FILING_REPORT_HUMAN = """Company: {company_name} | Filing: {filing_type}

=== Risk Analysis ===
{risk_analysis}

=== Sentiment Analysis ===
{sentiment_analysis}

=== Fundamental Analysis ===
{fundamental_analysis}

=== Mathematical Computations ===
{math_results}

Generate the 2-page filing summary."""


def generate_filing_report_node(state: SECState) -> dict:
    if state.get("error"):
        return {
            "filing_report": f"## Filing Analysis Failed\n\nError: {state.get('error', 'Unknown error')}"
        }

    llm = ChatGroq(
        model=settings.primary_model,
        temperature=0.1,
        api_key=settings.groq_api_key,
    )

    human = _FILING_REPORT_HUMAN.format(
        company_name=state.get("company_name", state["ticker"]),
        filing_type=state.get("filing_type", "10-K"),
        risk_analysis=state.get("risk_analysis", "Not available"),
        sentiment_analysis=state.get("sentiment_analysis", "Not available"),
        fundamental_analysis=state.get("fundamental_analysis", "Not available"),
        math_results=str(state.get("math_results", {})),
    )

    response = llm.invoke([
        SystemMessage(content=_FILING_REPORT_SYSTEM),
        HumanMessage(content=human),
    ])

    return {"filing_report": response.content}


# ---------------------------------------------------------------------------
# Build compiled graph
# ---------------------------------------------------------------------------

def build_sec_graph():
    g = StateGraph(SECState)

    g.add_node("resolve_cik", resolve_cik_node)
    g.add_node("download_filing", download_filing_node)
    g.add_node("chunk_filing", chunk_filing_node)
    g.add_node("enrich_chunks", enrich_chunks_node)
    g.add_node("build_index", build_index_node)
    g.add_node("risk_analyst", risk_analyst_node)
    g.add_node("sentiment_analyst", sentiment_analyst_node)
    g.add_node("fundamental_analyst", fundamental_analyst_node)
    g.add_node("math_agent", math_agent_node)
    g.add_node("generate_filing_report", generate_filing_report_node)

    g.set_entry_point("resolve_cik")
    g.add_edge("resolve_cik", "download_filing")
    g.add_edge("download_filing", "chunk_filing")
    g.add_edge("chunk_filing", "enrich_chunks")
    g.add_edge("enrich_chunks", "build_index")
    g.add_conditional_edges(
        "build_index",
        fan_out_experts,
        ["risk_analyst", "sentiment_analyst", "fundamental_analyst", "math_agent"],
    )
    g.add_edge("risk_analyst", "generate_filing_report")
    g.add_edge("sentiment_analyst", "generate_filing_report")
    g.add_edge("fundamental_analyst", "generate_filing_report")
    g.add_edge("math_agent", "generate_filing_report")
    g.add_edge("generate_filing_report", END)

    return g.compile()
