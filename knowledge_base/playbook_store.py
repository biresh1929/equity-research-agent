"""ChromaDB + BM25 store for AI playbook (institutional memory)."""

import json
import logging
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from rank_bm25 import BM25Okapi

from config.settings import settings
from .models import PlaybookEntry

logger = logging.getLogger(__name__)

COLLECTION_NAME = "playbooks"


class PlaybookStore:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=DefaultEmbeddingFunction(),
        )
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_entries: list[PlaybookEntry] = []
        self._bm25_dirty = True

    def _searchable_text(self, entry: PlaybookEntry) -> str:
        parts = [
            entry.research_summary,
            entry.bull_thesis,
            entry.bear_thesis,
        ] + entry.key_risks
        return "\n".join(p for p in parts if p)

    def add_entry(self, entry: PlaybookEntry) -> None:
        text = self._searchable_text(entry)
        self._collection.upsert(
            ids=[entry.id],
            documents=[text],
            metadatas=[{
                "ticker": entry.ticker,
                "company_name": entry.company_name,
                "sector": entry.sector,
                "decision": entry.decision,
                "conviction": entry.conviction,
                "date": entry.date.isoformat(),
                "health_score": entry.health_score,
                "growth_score": entry.growth_score,
                "mode": entry.mode,
                "user_feedback": entry.user_feedback or "",
                # Store full entry as JSON for reconstruction
                "_json": entry.model_dump_json(),
            }],
        )
        self._bm25_dirty = True
        logger.info("Saved playbook entry %s", entry.id)

    def update_feedback(self, entry_id: str, feedback: str, notes: str = "") -> None:
        results = self._collection.get(ids=[entry_id], include=["metadatas", "documents"])
        if not results["ids"]:
            logger.warning("Playbook entry %s not found", entry_id)
            return

        meta = results["metadatas"][0]
        try:
            entry = PlaybookEntry.model_validate_json(meta["_json"])
        except Exception:
            return

        entry.user_feedback = feedback
        entry.feedback_notes = notes
        self.add_entry(entry)

    def _ensure_bm25(self) -> None:
        if not self._bm25_dirty:
            return
        all_entries = self._get_all_entries()
        if not all_entries:
            self._bm25 = None
            return
        self._bm25_entries = all_entries
        tokenized = [self._searchable_text(e).lower().split() for e in all_entries]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_dirty = False

    def count(self) -> int:
        return self._collection.count()

    def get_feedback_summary(self) -> list[dict]:
        """Return id + ticker + decision + user_feedback for all entries (for inspection)."""
        entries = self._get_all_entries()
        return [
            {
                "id": e.id,
                "ticker": e.ticker,
                "decision": e.decision,
                "user_feedback": e.user_feedback,
            }
            for e in entries
        ]

    def _get_all_entries(self) -> list[PlaybookEntry]:
        try:
            count = self._collection.count()
            if count == 0:
                return []
            results = self._collection.get(include=["metadatas"], limit=count)
            entries = []
            for meta in results.get("metadatas", []):
                try:
                    entry = PlaybookEntry.model_validate_json(meta["_json"])
                    entries.append(entry)
                except Exception:
                    pass
            return entries
        except Exception:
            return []

    def query_similar(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        sector: Optional[str] = None,
        n_results: int = 5,
    ) -> list[PlaybookEntry]:
        """
        Hybrid search: semantic + optional metadata filter.
        Priority: same ticker > same sector > general.
        """
        count = self._collection.count()
        if count == 0:
            return []

        where_filter: Optional[dict] = None
        if ticker:
            where_filter = {"ticker": {"$eq": ticker}}
        elif sector and sector:
            where_filter = {"sector": {"$eq": sector}}

        kwargs: dict = {
            "query_texts": [query_text],
            "n_results": min(n_results, count),
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self._collection.query(**kwargs)

        entries = []
        for meta in (results.get("metadatas") or [[]])[0]:
            try:
                entry = PlaybookEntry.model_validate_json(meta["_json"])
                entries.append(entry)
            except Exception:
                pass

        return entries
