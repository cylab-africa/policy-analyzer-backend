"""Security checks: no pickle in the Redis cache, and the
API docs / OpenAPI file being off by default."""
import importlib
# pickle is imported on purpose, only to BUILD legacy and malicious payloads that the cache
# must refuse to load. The code under test never unpickles anything.
import pickle
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services import query_cache
from app.services.query_cache import QueryCache


class FakeRedis:
    """Just enough of the Redis client for QueryCache."""

    def __init__(self):
        self.kv = {}
        self.hashes = {}

    def exists(self, key):
        return key in self.kv or key in self.hashes

    def get(self, key):
        return self.kv.get(key)

    def setex(self, key, _ttl, value):
        self.kv[key] = value

    def hset(self, name, key=None, value=None, mapping=None):
        h = self.hashes.setdefault(name, {})
        if mapping:
            h.update({k.encode() if isinstance(k, str) else k: v for k, v in mapping.items()})
        if key is not None:
            h[key.encode() if isinstance(key, str) else key] = value

    def hgetall(self, name):
        return dict(self.hashes.get(name, {}))

    def hdel(self, name, *keys):
        for key in keys:
            self.hashes.get(name, {}).pop(key.encode() if isinstance(key, str) else key, None)

    def pipeline(self, transaction=True):
        fake, keys = self, []

        class Pipeline:
            def exists(self, key):
                keys.append(key)

            def execute(self):
                return [int(fake.exists(k)) for k in keys]

        return Pipeline()

    def hincrby(self, name, key, amount):
        h = self.hashes.setdefault(name, {})
        k = key.encode()
        h[k] = int(h.get(k, 0)) + amount

    def hlen(self, name):
        return len(self.hashes.get(name, {}))

    def expire(self, name, ttl):
        pass

    def delete(self, *keys):
        for k in keys:
            self.kv.pop(k, None)
            self.hashes.pop(k, None)

    def scan(self, cursor, match=None, count=None):
        prefix = (match or "").rstrip("*")
        return 0, [k for k in self.kv if k.startswith(prefix)]


RESULTS = [{"content": "Kenya Data Protection Act", "country": "Kenya", "year": 2019}]


@pytest.fixture()
def cache():
    return QueryCache(FakeRedis(), similarity_threshold=0.98)


def test_exact_match_round_trips_through_json(cache):
    embedding = np.array([0.1, 0.2, 0.3])
    cache.set("What is the law?", embedding, {"tab": "dp"}, RESULTS)
    assert cache.get("what is the law?", embedding, {"tab": "dp"}) == RESULTS


def test_similar_query_is_found_by_embedding(cache):
    cache.set("What is the law?", np.array([1.0, 0.0, 0.0]), {}, RESULTS)
    nearly_same = np.array([1.0, 0.001, 0.0])
    assert cache.get("A differently worded question", nearly_same, {}) == RESULTS
    assert cache.get("Unrelated", np.array([0.0, 1.0, 0.0]), {}) is None


def test_stored_values_are_json_not_pickle(cache):
    cache.set("q", np.array([0.5, 0.5]), {}, RESULTS)
    redis = cache.redis
    values = list(redis.kv.values()) + list(redis.hashes[cache.embeddings_key_for({})].values())
    assert values
    for raw in values:
        assert raw.startswith(b"{")  # pickle data starts with a protocol marker, never "{"


def test_legacy_pickled_entries_are_a_miss_and_get_cleaned_up(cache):
    redis = cache.redis
    redis.setex(f"{cache.cache_key_prefix}abc", 60, pickle.dumps({"results": RESULTS, "query": "q"}))
    redis.hset(cache.embeddings_key_for({}), "abc",
               pickle.dumps({"embedding": np.array([1.0, 0.0]), "filter_str": "{}"}))
    assert cache.get("q", np.array([1.0, 0.0]), {}) is None
    assert b"abc" not in redis.hgetall(cache.embeddings_key_for({}))


def test_a_poisoned_pickle_payload_is_never_executed(cache, tmp_path):
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (marker.write_text, ("x",))

    cache.redis.setex(f"{cache.cache_key_prefix}{cache._create_cache_key('q', {})}", 60, pickle.dumps(Evil()))
    assert cache.get("q", np.array([1.0, 0.0]), {}) is None
    assert not marker.exists()


def test_query_cache_module_does_not_import_pickle():
    source = Path(query_cache.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import pickle|from pickle)", source, re.MULTILINE)
    assert "pickle.loads" not in source


# --- API docs ------------------------------------------------------------------------------

@pytest.fixture()
def load_app(monkeypatch):
    import main

    def build(enabled: bool):
        monkeypatch.setattr(settings, "enable_api_docs", enabled)
        return TestClient(importlib.reload(main).app)

    yield build
    monkeypatch.setattr(settings, "enable_api_docs", False)
    importlib.reload(main)


def test_docs_and_openapi_are_off_by_default(load_app):
    client = load_app(False)
    for path in ("/api/docs", "/openapi.json", "/redoc", "/docs"):
        assert client.get(path).status_code == 401, path  # no key: rejected, not served
        assert client.get(path, headers={"X-API-Key": settings.api_key}).status_code == 404, path


def test_docs_can_be_enabled_for_local_development(load_app):
    client = load_app(True)
    assert client.get("/api/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert "/api/query_v2" in client.get("/openapi.json").json()["paths"]
