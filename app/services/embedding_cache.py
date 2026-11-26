"""
Remember recent query embeddings so the same question is not embedded (and paid for) twice.

A search embeds the question once for the query-cache lookup and, on a miss, the vector store
embeds the same text again inside similarity_search. Qdrant runs in hybrid mode (dense plus
sparse vectors), so passing a ready-made vector to similarity_search_by_vector would silently
drop the sparse half. Wrapping the embedding model instead keeps the search unchanged and turns
the second call into a lookup.
"""
import threading
from collections import OrderedDict
from typing import List, Tuple

from langchain_core.embeddings import Embeddings


class CachingQueryEmbeddings(Embeddings):
    """Delegates to ``inner``; ``embed_query`` results are kept for the most recent texts."""

    def __init__(self, inner: Embeddings, max_entries: int = 256):
        self.inner = inner
        self.max_entries = max_entries
        self._cache: "OrderedDict[str, Tuple[float, ...]]" = OrderedDict()
        self._lock = threading.Lock()

    def embed_query(self, text: str) -> List[float]:
        with self._lock:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                return list(cached)

        vector = self.inner.embed_query(text)

        with self._lock:
            self._cache[text] = tuple(vector)
            self._cache.move_to_end(text)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
        return list(vector)

    async def aembed_query(self, text: str) -> List[float]:
        with self._lock:
            cached = self._cache.get(text)
        if cached is not None:
            return list(cached)
        return self.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Documents are embedded once, at ingestion; nothing to reuse
        return self.inner.embed_documents(texts)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self.inner.aembed_documents(texts)

    def __getattr__(self, name):
        # Anything else (model name, dimensions, ...) comes from the wrapped model
        if name == "inner":  # not set yet (e.g. while copying): avoid endless recursion
            raise AttributeError(name)
        return getattr(self.inner, name)


def with_query_cache(embeddings):
    """Wrap ``embeddings`` once; already-wrapped or missing models are returned as they are."""
    if embeddings is None or isinstance(embeddings, CachingQueryEmbeddings):
        return embeddings
    return CachingQueryEmbeddings(embeddings)
