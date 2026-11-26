"""The same question is embedded once per search, not twice (cache lookup, then the search)."""
from langchain_core.embeddings import Embeddings

from app.services.embedding_cache import CachingQueryEmbeddings, with_query_cache


class CountingEmbeddings(Embeddings):
    model = "test-model"

    def __init__(self):
        self.query_calls = 0
        self.document_calls = 0

    def embed_query(self, text):
        self.query_calls += 1
        return [float(len(text)), 1.0]

    def embed_documents(self, texts):
        self.document_calls += 1
        return [[float(len(t)), 1.0] for t in texts]


def test_repeated_query_is_embedded_once():
    inner = CountingEmbeddings()
    cached = CachingQueryEmbeddings(inner)
    first = cached.embed_query("What is the law?")
    second = cached.embed_query("What is the law?")
    assert first == second == [16.0, 1.0]
    assert inner.query_calls == 1
    cached.embed_query("Another question")
    assert inner.query_calls == 2


def test_results_cannot_be_changed_through_a_returned_list():
    cached = CachingQueryEmbeddings(CountingEmbeddings())
    cached.embed_query("q").append(99.0)
    assert cached.embed_query("q") == [1.0, 1.0]


def test_only_recent_queries_are_kept():
    inner = CountingEmbeddings()
    cached = CachingQueryEmbeddings(inner, max_entries=2)
    for text in ("a", "b", "c", "a"):
        cached.embed_query(text)
    assert inner.query_calls == 4  # "a" had been pushed out by "b" and "c"


def test_documents_and_attributes_pass_through():
    inner = CountingEmbeddings()
    cached = with_query_cache(inner)
    assert cached.embed_documents(["x", "yy"]) == [[1.0, 1.0], [2.0, 1.0]]
    assert inner.document_calls == 1
    assert cached.model == "test-model"
    assert with_query_cache(cached) is cached
    assert with_query_cache(None) is None
