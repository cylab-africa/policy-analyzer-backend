"""Fixes from the services review: private-document scoping, vector store operations, keeping
blocking calls off the event loop, per-request log context, provider health from real calls,
prompt substitution and email escaping. No model, database or network is used."""
import asyncio
import threading
import time

import pytest

from app.services import llm_factory, query_handler as qh, vector_store as vs


# --- private documents keep their scope --------------------------------------------------------

class RecordingStore(vs.VectorStoreV2):
    """Finds nothing, and records the filters of every search."""

    def __init__(self):
        self.searched = []

    def search(self, query, filter, top_k=5):
        self.searched.append(dict(filter))
        return []

    add_documents = delete_documents = update_document = clear_store = get_vector_store = None


def test_the_last_fallback_keeps_the_client_id():
    store = RecordingStore()
    store.search_with_filter_combinations(
        "q", [{"client_id": "client-1", "country": "Kenya", "tab": "data-protection"}], top_k=3
    )
    assert store.searched, "searched at least once"
    assert all(f.get("client_id") == "client-1" for f in store.searched)
    assert store.searched[-1] == {"client_id": "client-1"}


def test_the_last_fallback_without_a_client_searches_everything():
    store = RecordingStore()
    store.search_with_filter_combinations("q", [{"country": "Kenya"}], top_k=3)
    assert store.searched[-1] == {}


class FakeCache:
    def __init__(self):
        self.lookups = []

    def get(self, query, embedding, filters):
        self.lookups.append(filters)
        return None

    def set(self, query, embedding, filters, results):
        pass


class FakeEmbeddings:
    def embed_query(self, text):
        return [0.1, 0.2]


class FakeLangchainSearch:
    def __init__(self):
        self.calls = []

    def similarity_search(self, query, **kwargs):
        self.calls.append(kwargs)
        return []


def qdrant_store(name):
    store = object.__new__(vs.QdrantVectorStore)  # no Qdrant connection
    store._collection_name = name
    store._embedding_function = FakeEmbeddings()
    store._vector_store = FakeLangchainSearch()
    return store


def test_public_and_private_searches_are_cached_apart(monkeypatch):
    cache = FakeCache()
    monkeypatch.setattr(vs, "get_query_cache_or_none", lambda: cache)
    qdrant_store("public_docs").search("q", {"tab": "x"})
    qdrant_store("private_docs").search("q", {"tab": "x"})
    first, second = cache.lookups
    assert first != second
    assert first["__store__"] == "public_docs" and second["__store__"] == "private_docs"
    assert first["tab"] == second["tab"] == "x"


# --- vector store operations -------------------------------------------------------------------

def test_qdrant_filter_uses_metadata_fields_and_drops_query_only_keys():
    store = qdrant_store("docs")
    f = store._to_qdrant_filter({"country": ["Kenya", "Uganda"], "tab": "dp", "is_comparative": True})
    assert sorted(c.key for c in f.must) == ["metadata.country", "metadata.tab"]
    assert {c.match.value for c in f.must} == {"Kenya", "dp"}
    assert store._to_qdrant_filter({"is_broad": True}) is None


class FakeQdrantClient:
    def __init__(self):
        self.deleted = []
        self.dropped = []

    def delete(self, collection_name, points_selector):
        self.deleted.append((collection_name, points_selector))

    def delete_collection(self, name):
        self.dropped.append(name)


def test_qdrant_delete_by_filter_and_refuses_an_empty_filter():
    store = qdrant_store("docs")
    store._client = FakeQdrantClient()
    store.delete_documents({"client_id": "client-1"})
    (name, selector), = store._client.deleted
    assert name == "docs"
    assert selector.filter.must[0].key == "metadata.client_id"
    with pytest.raises(ValueError):
        store.delete_documents({})
    assert len(store._client.deleted) == 1


def test_qdrant_clear_drops_and_recreates_the_collection(monkeypatch):
    store = qdrant_store("docs")
    store._client = FakeQdrantClient()
    recreated = []
    monkeypatch.setattr(store, "_ensure_collection_exists", lambda: recreated.append("ensure"))
    monkeypatch.setattr(store, "_init_vector_store", lambda: recreated.append("init"))
    store.clear_store()
    assert store._client.dropped == ["docs"]
    assert recreated == ["ensure", "init"]


def test_updating_a_chunk_says_it_is_not_supported():
    with pytest.raises(NotImplementedError):
        qdrant_store("docs").update_document("id", {})
    with pytest.raises(NotImplementedError):
        object.__new__(vs.MongoDBVectorStore).update_document("id", {})


class FakeMongoCollection:
    name = "mongo_docs"

    def __init__(self):
        self.deleted = []

    def delete_many(self, query):
        self.deleted.append(query)


def test_mongodb_search_uses_a_mongodb_pre_filter(monkeypatch):
    monkeypatch.setattr(vs, "get_query_cache_or_none", lambda: None)
    store = object.__new__(vs.MongoDBVectorStore)
    store._vector_store = FakeLangchainSearch()
    store.search("q", {"country": ["Kenya"], "tab": "dp", "regions": ["East"]}, top_k=4)
    (call,) = store._vector_store.calls
    assert call == {"k": 4, "pre_filter": {"country": {"$eq": "Kenya"}, "tab": {"$eq": "dp"}}}


def test_mongodb_delete_by_filter_and_refuses_an_empty_filter():
    store = object.__new__(vs.MongoDBVectorStore)
    store._vector_store = type("S", (), {"collection": FakeMongoCollection()})()
    store.delete_documents({"client_id": ["client-1"]})
    assert store._vector_store.collection.deleted == [{"client_id": "client-1"}]
    with pytest.raises(ValueError):
        store.delete_documents({"is_comparative": True})


def test_adding_documents_reports_failure():
    store = qdrant_store("docs")

    class Failing:
        def add_documents(self, docs):
            raise RuntimeError("qdrant unavailable")

    store._vector_store = Failing()
    with pytest.raises(RuntimeError):
        store.add_documents([{"content": "text"}])


# --- blocking work stays off the event loop ----------------------------------------------------

class SlowAsyncLLM:
    """Each call takes 0.3 s without blocking the loop; the blocking form must not be used."""

    def __init__(self, answer):
        self.answer = answer

    async def ainvoke(self, prompt):
        await asyncio.sleep(0.3)
        return type("R", (), {"content": self.answer})()

    def invoke(self, prompt):
        raise AssertionError("the blocking invoke() was called from async code")


class FakePromptManager:
    def get_prompt(self, key, variables=None):
        return f"{key}: {variables}"


@pytest.fixture()
def handler(monkeypatch):
    h = qh.DynamicQueryHandler()
    h._prompt_manager = FakePromptManager()
    monkeypatch.setattr(qh.LLMFactory, "create_llm", lambda **_: SlowAsyncLLM("RELEVANT"))
    # The tab list comes from the database; these tests are about the model calls
    monkeypatch.setattr(type(qh.settings), "get_enabled_tabs", lambda self: ["data-protection"])
    return h


def test_the_three_checks_run_at_the_same_time(handler):
    history = [{"role": "human", "content": "Tell me about Kenya"}]

    async def run():
        start = time.perf_counter()
        await asyncio.gather(
            handler._enhance_query_with_context("Is it mandatory?", history),
            handler._is_query_relevant_to_domain("Is it mandatory?", history),
            handler.detect_tab("Is it mandatory?", history),
        )
        return time.perf_counter() - start

    # One after another they would take 0.9 s
    assert asyncio.run(run()) < 0.75


def test_retrieval_and_web_search_run_in_worker_threads(handler, monkeypatch):
    threads = {}

    class Store:
        def search_with_metadata(self, query, tab, metadata_filters):
            threads["search"] = threading.current_thread()
            return []

    handler._vector_stores = {"private": Store()}

    def search(**_):
        threads["web"] = threading.current_thread()
        return "results"

    from app.services.search_tools import SearchToolFactory
    monkeypatch.setattr(SearchToolFactory, "perform_search", staticmethod(search))

    async def run():
        await handler.retrieve_documents("q", visibility="private",
                                         metadata_filters={"client_id": "c"}, use_auto_filters=False)
        await handler.perform_web_search("q")
        return threading.current_thread()

    loop_thread = asyncio.run(run())
    assert threads["search"] is not loop_thread
    assert threads["web"] is not loop_thread


def test_entities_are_detected_once_per_question(monkeypatch):
    from app.services.entity_detector import EntityDetector

    detector = EntityDetector(llm=None)
    calls = []

    def analyse(query, config):
        calls.append(query)
        return {"country": ["Kenya"], "is_comparative": False}

    monkeypatch.setattr(detector, "_detect_all_entities_uncached", analyse)
    first = detector.detect_all_entities("Kenya's law?", {})
    first["country"].append("changed by a caller")
    second = detector.detect_all_entities("Kenya's law?", {})
    assert calls == ["Kenya's law?"]
    assert second["country"] == ["Kenya"]
    detector.detect_all_entities("Uganda's law?", {})
    assert calls == ["Kenya's law?", "Uganda's law?"]


def test_cached_entities_expire(monkeypatch):
    from app.services.entity_detector import EntityDetector

    detector = EntityDetector(llm=None)
    calls = []
    monkeypatch.setattr(detector, "_detect_all_entities_uncached", lambda q, c: calls.append(q) or {})
    detector._entities_cache_seconds = 0
    detector.detect_all_entities("q", {})
    detector.detect_all_entities("q", {})
    assert len(calls) == 2


# --- log context per request -------------------------------------------------------------------

def test_concurrent_requests_keep_their_own_log_context():
    logger = qh.StructuredLogger("test_context")

    async def request(query_id, seen):
        logger.set_context(query_id=query_id)
        await asyncio.sleep(0.01)  # the other request sets its context meanwhile
        seen[query_id] = logger._format_message("done")
        logger.clear_context()

    async def run():
        seen = {}
        await asyncio.gather(request("A", seen), request("B", seen))
        return seen

    seen = asyncio.run(run())
    assert seen == {"A": "done | query_id=A", "B": "done | query_id=B"}


# --- provider health from real calls -----------------------------------------------------------

@pytest.fixture()
def breaker(monkeypatch):
    fresh = llm_factory.CircuitBreaker(failure_threshold=3, recovery_timeout=60)
    monkeypatch.setattr(llm_factory, "_circuit_breaker", fresh)
    monkeypatch.setattr(llm_factory, "_llm_metrics", llm_factory.LLMCallMetrics())
    return fresh


def test_failed_calls_open_the_circuit_and_successes_are_counted(breaker):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    class FailingModel(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise RuntimeError("provider down")

    callback = llm_factory.ProviderHealthCallback("testprov", "query_response")
    ok = FakeListChatModel(responses=["fine"], callbacks=[callback])
    assert ok.invoke("hi").content == "fine"
    assert asyncio.run(ok.ainvoke("hi")).content == "fine"
    stats = llm_factory.get_llm_metrics().get_stats()["testprov:query_response"]
    assert stats["total_calls"] == 2 and stats["error_count"] == 0

    failing = FailingModel(responses=["x"], callbacks=[callback])
    for _ in range(3):
        with pytest.raises(RuntimeError):
            failing.invoke("hi")
    assert breaker.get_state("testprov") == breaker.OPEN
    assert not breaker.can_execute("testprov")


def test_creating_a_model_does_not_count_as_a_healthy_call(breaker):
    llm = llm_factory.LLMFactory.create_llm(use_case="query_response")
    assert any(isinstance(c, llm_factory.ProviderHealthCallback) for c in llm.callbacks)
    breaker.record_failure("openai")
    breaker.record_failure("openai")
    llm_factory.LLMFactory.create_llm(use_case="query_response")
    assert breaker._failure_counts["openai"] == 2  # not reset by construction


def test_models_get_a_timeout_and_client_retries():
    kwargs = llm_factory.LLMFactory._build_llm_kwargs("openai", "m", {}, None)
    assert kwargs["timeout"] == llm_factory.DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert kwargs["max_retries"] == llm_factory.DEFAULT_MAX_RETRIES
    configured = llm_factory.LLMFactory._build_llm_kwargs("openai", "m", {}, {"request_timeout": 5})
    assert configured["request_timeout"] == 5 and "timeout" not in configured
    assert "timeout" not in llm_factory.LLMFactory._build_llm_kwargs("ollama", "m", {}, None)


def test_created_model_really_has_the_timeout():
    llm = llm_factory.LLMFactory.create_llm(use_case="query_response")
    assert llm.request_timeout is not None
    assert llm.max_retries == llm_factory.DEFAULT_MAX_RETRIES


# --- prompt substitution -----------------------------------------------------------------------

@pytest.fixture()
def prompts():
    from app.services.prompt_manager import PromptManager

    return PromptManager(db_client=None)


def test_a_missing_variable_leaves_only_its_placeholder(prompts):
    out = prompts._substitute_variables("Q: {question}\nC: {context}", {"question": "Why?"})
    assert out == "Q: Why?\nC: {context}"


def test_braces_that_are_not_placeholders_do_not_stop_substitution(prompts):
    template = 'Answer {question} as JSON like {"country": "x"}'
    out = prompts._substitute_variables(template, {"question": "Why?"})
    assert out == 'Answer Why? as JSON like {"country": "x"}'


def test_values_are_inserted_as_they_are(prompts):
    out = prompts._substitute_variables("{context}", {"context": "a {b} c"})
    assert out == "a {b} c"


# --- email -------------------------------------------------------------------------------------

def test_completion_email_escapes_its_values():
    from datetime import datetime

    from app.services.email_service import EmailService

    service = object.__new__(EmailService)
    html = service._format_html_email(
        "user@example.com",
        {"job_id": "<script>x</script>", "completed_at": datetime(2026, 1, 1)},
        pdf_download_url='https://s/r.pdf?a=1&b="><script>',
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert 'href="https://s/r.pdf?a=1&amp;b=&quot;&gt;&lt;script&gt;"' in html


def test_failure_email_escapes_and_keeps_headers_on_one_line(monkeypatch):
    from app.services.email_service import EmailService

    sent = {}

    class FakeServer:
        def sendmail(self, sender, recipient, message):
            sent["message"] = message

        def quit(self):
            pass

    service = object.__new__(EmailService)
    service.smtp_username, service.smtp_password = "user", "pass"
    service.sender_email, service.sender_name = "noreply@example.com", "PDP"
    monkeypatch.setattr(service, "_create_connection", lambda: FakeServer())
    assert service.send_batch_failed_email(
        "a@example.com\nBcc: everyone@example.com", "job\nX-Injected: 1", "<b>bad</b>"
    )
    message = sent["message"]
    headers = message.split("\n\n", 1)[0]
    assert "\nBcc:" not in headers and "\nX-Injected:" not in headers
    assert "<b>bad</b>" not in message.split("text/html", 1)[1]


# --- tab list is not read from the database for every call --------------------------------------

def test_the_tab_list_is_cached_and_cleared_on_change():
    from app.services.tab_manager import TabManager

    class Collection:
        def __init__(self):
            self.finds = 0

        def find(self):
            self.finds += 1
            return [{"tab_name": "extra", "config": {"display_name": "X", "description": "x"}}]

        def update_one(self, *args, **kwargs):
            pass

    collection = Collection()
    db = type("Db", (), {"get_collection": lambda self, name: collection})()
    manager = TabManager(db_client=db)
    for _ in range(5):
        tabs = manager.list_all_tabs()
    assert collection.finds == 1
    assert "extra" in {t["tab_name"] for t in tabs}

    tabs[0]["config"]["changed"] = True  # a caller changing its copy does not change the cache
    assert "changed" not in manager.list_all_tabs()[0]["config"]

    manager.save_tab_override("extra", {"display_name": "X", "description": "y"})
    manager.list_all_tabs()
    assert collection.finds == 2
