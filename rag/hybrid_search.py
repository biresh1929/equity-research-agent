"""Reciprocal Rank Fusion for hybrid ChromaDB + BM25 search."""

from .chroma_store import ChromaStore
from .bm25_store import get_store


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def hybrid_search(
    collection_name: str,
    query: str,
    n_results: int = 8,
    alpha: float | None = None,
) -> list[dict]:
    """
    Hybrid search using Reciprocal Rank Fusion.

    alpha controls weight split (default from settings):
      1.0 = semantic only, 0.0 = BM25 only, 0.6 = default blend.
    """
    from config.settings import settings
    if alpha is None:
        alpha = settings.hybrid_alpha

    chroma = ChromaStore(collection_name)
    bm25 = get_store(collection_name)

    semantic_results = chroma.query(query, n_results=n_results * 2)
    bm25_results = bm25.query(query, n_results=n_results * 2) if bm25 else []

    # Build RRF score map keyed by chunk_id
    scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for rank, result in enumerate(semantic_results):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + alpha * _rrf_score(rank)
        chunk_data[cid] = result

    for rank, result in enumerate(bm25_results):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + (1.0 - alpha) * _rrf_score(rank)
        if cid not in chunk_data:
            chunk_data[cid] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
    return [
        {**chunk_data[cid], "hybrid_score": score}
        for cid, score in ranked
    ]
