"""ChromaDB store with local embeddings (all-MiniLM-L6-v2, no API key)."""

import re
import logging
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from config.settings import settings

logger = logging.getLogger(__name__)


def safe_collection_name(ticker: str, filing_type: str, cik: str) -> str:
    """ChromaDB names: 3–63 chars, [a-zA-Z0-9_-] only."""
    raw = f"{ticker}_{filing_type}_{cik}".lower()
    safe = re.sub(r"[^a-z0-9_-]", "_", raw)
    # Ensure at least 3 chars
    safe = safe.ljust(3, "_")
    return safe[:63]


class ChromaStore:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=DefaultEmbeddingFunction(),
        )

    def upsert_chunks(self, chunks: list[dict]) -> None:
        """Upsert chunks into the collection. Uses chunk_id as document ID."""
        if not chunks:
            return

        ids = [c["chunk_id"] for c in chunks]
        # Combine text + description for richer embeddings
        documents = [
            f"{c.get('description', '')}\n{c['text']}" for c in chunks
        ]
        metadatas = [
            {
                "doc_id": c["doc_id"],
                "section": c.get("section", ""),
                "is_table": str(c.get("is_table", False)),
                "company": c.get("company", ""),
                "filing_type": c.get("filing_type", ""),
                "year": str(c.get("year", "")),
            }
            for c in chunks
        ]

        # ChromaDB upsert in batches of 100
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self._collection.upsert(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
        logger.info("Upserted %d chunks into collection %s", len(ids), self.collection_name)

    def query(
        self,
        query_text: str,
        n_results: int = 8,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Semantic search. Returns list of {chunk_id, text, score, metadata}.
        """
        kwargs: dict = {"query_texts": [query_text], "n_results": min(n_results, self._collection.count() or 1)}
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results.get("distances") else [1.0] * len(docs)

        output = []
        for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
            output.append({
                "chunk_id": doc_id,
                "text": doc,
                "score": 1.0 - dist,  # convert distance to similarity
                "metadata": meta,
            })
        return output

    def count(self) -> int:
        return self._collection.count()

    def get_all_chunks(self) -> list[dict]:
        """Return all stored documents as chunk dicts (used to rebuild BM25 on restart)."""
        total = self._collection.count()
        if total == 0:
            return []
        result = self._collection.get(limit=total, include=["documents", "metadatas"])
        chunks = []
        for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            chunks.append({
                "chunk_id": cid,
                "text": doc,
                "section": meta.get("section", ""),
                "is_table": meta.get("is_table", "False") == "True",
                "company": meta.get("company", ""),
                "filing_type": meta.get("filing_type", ""),
                "year": meta.get("year", ""),
            })
        return chunks
