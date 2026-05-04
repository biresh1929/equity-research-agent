"""BM25 index with module-level registry to avoid LangGraph state serialization issues."""

import logging
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Module-level registry — state only stores the collection_name string
_REGISTRY: dict[str, "BM25Store"] = {}


class BM25Store:
    def __init__(self):
        self._index: BM25Okapi | None = None
        self._chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        self._chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self._index = BM25Okapi(tokenized)
        logger.info("Built BM25 index with %d chunks", len(chunks))

    def query(self, query_text: str, n_results: int = 8) -> list[dict]:
        if not self._index or not self._chunks:
            return []

        tokens = query_text.lower().split()
        scores = self._index.get_scores(tokens)

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:n_results]

        return [
            {
                "chunk_id": self._chunks[i]["chunk_id"],
                "text": self._chunks[i]["text"],
                "score": float(score),
                "metadata": {
                    "section": self._chunks[i].get("section", ""),
                    "is_table": str(self._chunks[i].get("is_table", False)),
                },
            }
            for i, score in ranked
            if score > 0
        ]


def get_or_build(collection_name: str, chunks: list[dict]) -> BM25Store:
    """Return existing BM25Store for collection_name, or build from chunks."""
    if collection_name not in _REGISTRY:
        store = BM25Store()
        store.build(chunks)
        _REGISTRY[collection_name] = store
        logger.info("Registered BM25Store for %s", collection_name)
    return _REGISTRY[collection_name]


def get_store(collection_name: str) -> BM25Store | None:
    return _REGISTRY.get(collection_name)


def clear_registry() -> None:
    """Clear all stores (useful between test runs)."""
    _REGISTRY.clear()
