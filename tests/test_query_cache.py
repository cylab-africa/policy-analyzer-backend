"""Query cache: values keep their types through Redis, and Redis trouble never breaks a search."""
from datetime import date, datetime, timezone
from decimal import Decimal

import numpy as np
import pytest
import redis
from bson import ObjectId

from app.services.query_cache import QueryCache, _dumps, _loads


class FakeRedis:
    """Just enough of redis.Redis for QueryCache. Operations named in ``failing`` raise."""

    def __init__(self):
        self.values = {}
        self.hashes = {}
        self.failing = set()
        # Unreachable server by default; redis.ResponseError for an error reply (READONLY, OOM)
        self.error = redis.ConnectionError
        self.round_trips = 0
        self.calls = 0
        self.expiries = {}

    def __getattribute__(self, name):
        # Count every Redis command, to check that nothing is sent while Redis is known down
        if name in {"get", "setex", "hset", "hgetall", "hdel", "hincrby", "pipeline"}:
            object.__setattr__(self, "calls", object.__getattribute__(self, "calls") + 1)
        return object.__getattribute__(self, name)

    def _maybe_fail(self, name):
        if name in self.failing:
            raise self.error(f"{name} failed")

    def exists(self, key):
        return key in self.values or key in self.hashes

    def get(self, key):
        self._maybe_fail("get")
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self._maybe_fail("setex")
        self.values[key] = value

    def hset(self, key, field=None, value=None, mapping=None):
        self._maybe_fail("hset")
        h = self.hashes.setdefault(key, {})
        if mapping:
            h.update({k.encode() if isinstance(k, str) else k: v for k, v in mapping.items()})
        else:
            h[field.encode() if isinstance(field, str) else field] = value

    def hgetall(self, key):
        self._maybe_fail("hgetall")
        return dict(self.hashes.get(key, {}))

    def hdel(self, key, *fields):
        self._maybe_fail("hdel")
        for field in fields:
            self.hashes.get(key, {}).pop(field.encode() if isinstance(field, str) else field, None)

    def pipeline(self, transaction=True):
        return FakePipeline(self)

    def delete(self, *keys):
        for key in keys:
            self.values.pop(key, None)
            self.hashes.pop(key, None)

    def expire(self, key, ttl):
        self.expiries[key] = ttl

    def hlen(self, key):
        return len(self.hashes.get(key, {}))

    def hincrby(self, key, field, amount):
        self._maybe_fail("hincrby")
        h = self.hashes.setdefault(key, {})
        k = field.encode()
        h[k] = int(h.get(k, 0)) + amount


class FakePipeline:
    def __init__(self, fake):
        self.fake, self.keys = fake, []

    def exists(self, key):
        self.keys.append(key)

    def execute(self):
        self.fake._maybe_fail("pipeline")
        self.fake.round_trips += 1
        return [int(self.fake.exists(k)) for k in self.keys]


EMBEDDING = np.array([0.1, 0.2, 0.3])
FILTERS = {"tab": "data-protection"}


@pytest.fixture()
def fake():
    return FakeRedis()


@pytest.fixture()
def cache(fake):
    return QueryCache(redis_client=fake)


def test_metadata_types_survive_a_round_trip():
    value = {
        "when": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        "day": date(2026, 1, 2),
        "amount": Decimal("12.50"),
        "id": ObjectId("65a1b2c3d4e5f6a7b8c9d0e1"),
        "tags": {"a", "b"},
        "vector": np.array([1.0, 2.0]),
        "plain": ["x", 1, None, True],
    }
    restored = _loads(_dumps(value))
    assert restored["when"] == value["when"]
    assert restored["day"] == value["day"] and type(restored["day"]) is date
    assert restored["amount"] == value["amount"] and isinstance(restored["amount"], Decimal)
    assert restored["id"] == value["id"]
    assert restored["tags"] == value["tags"]
    assert restored["vector"] == [1.0, 2.0]
    assert restored["plain"] == ["x", 1, None, True]


def test_unknown_types_are_refused_not_turned_into_strings():
    with pytest.raises(TypeError):
        _dumps({"x": object()})


def test_cache_hit_returns_the_same_types_as_the_original_search(cache):
    results = [{"content": "text", "published": datetime(2025, 5, 1), "id": ObjectId()}]
    cache.set("q", EMBEDDING, FILTERS, results)
    assert cache.get("q", EMBEDDING, FILTERS) == results


def test_results_that_cannot_be_stored_are_simply_not_cached(cache, fake):
    cache.set("q", EMBEDDING, FILTERS, [{"content": "text", "odd": object()}])
    assert not any(k.startswith(cache.cache_key_prefix) for k in fake.values)
    assert cache.get("q", EMBEDDING, FILTERS) is None


@pytest.mark.parametrize("failing", [{"setex"}, {"hincrby"}, {"setex", "hincrby"}])
def test_bookkeeping_failure_does_not_lose_an_exact_hit(cache, fake, failing):
    # A working server refusing a write (READONLY replica, out of memory)
    results = [{"content": "text"}]
    cache.set("q", EMBEDDING, FILTERS, results)
    fake.failing, fake.error = failing, redis.ResponseError
    assert cache.get("q", EMBEDDING, FILTERS) == results


def test_bookkeeping_failure_does_not_lose_a_similar_hit(cache, fake):
    results = [{"content": "text"}]
    cache.set("original question", EMBEDDING, FILTERS, results)
    fake.failing, fake.error = {"setex", "hincrby", "hdel"}, redis.ResponseError
    assert cache.get("a reworded question", EMBEDDING * 2, FILTERS) == results


def test_lost_connection_stops_the_lookup_at_once(cache, fake):
    from app.services import query_cache

    cache.set("q", EMBEDDING, FILTERS, [{"content": "text"}])
    fake.failing = {"hincrby"}  # the first command finds Redis unreachable
    fake.calls = 0
    assert cache.get("q", EMBEDDING, FILTERS) is None
    assert fake.calls == 1  # no further commands wait on the dead connection
    assert query_cache.redis_in_backoff()


def test_error_reply_does_not_start_the_backoff(cache, fake):
    from app.services import query_cache

    fake.failing, fake.error = {"get"}, redis.ResponseError
    assert cache.get("q", EMBEDDING, FILTERS) is None
    assert not query_cache.redis_in_backoff()


def test_redis_failure_while_storing_does_not_raise(cache, fake):
    fake.failing = {"setex", "hset"}
    cache.set("q", EMBEDDING, FILTERS, [{"content": "text"}])  # no exception


@pytest.mark.parametrize("failing", [{"get"}, {"hgetall"}, {"get", "hgetall", "hincrby"}])
def test_redis_read_failure_is_a_miss_not_an_error(cache, fake, failing):
    cache.set("original question", EMBEDDING, FILTERS, [{"content": "text"}])
    fake.failing = failing
    # A reworded question needs both the exact lookup (get) and the similarity scan (hgetall)
    assert cache.get("a reworded question", EMBEDDING, FILTERS) is None


@pytest.mark.parametrize("bad", [
    '{"__cache_type__": "decimal", "value": "abc"}',
    '{"__cache_type__": "objectid", "value": "not-hex"}',
    '{"__cache_type__": "datetime", "value": "yesterday"}',
    '{"__cache_type__": "set", "value": 5}',
])
def test_bad_typed_values_raise_value_error(bad):
    with pytest.raises(ValueError):
        _loads(bad)


def test_corrupt_typed_value_in_a_cached_entry_is_a_miss(cache, fake):
    key = cache.cache_key_prefix + cache._create_cache_key("q", FILTERS)
    fake.values[key] = b'{"results": [{"amount": {"__cache_type__": "decimal", "value": "abc"}}]}'
    assert cache.get("q", EMBEDDING, FILTERS) is None


def test_embeddings_are_stored_as_binary_not_a_number_list(cache, fake):
    cache.set("q", np.arange(1536, dtype=float), FILTERS, [{"content": "text"}])
    (stored,) = fake.hashes[cache.embeddings_key_for(FILTERS)].values()
    record = _loads(stored)
    assert "embedding" not in record and len(record["embedding_f32"]) < 1536 * 6


def test_embeddings_written_as_a_json_list_are_still_matched(cache, fake):
    import json

    cache.set("original question", EMBEDDING, FILTERS, [{"content": "text"}])
    (field,) = fake.hashes[cache.embeddings_key_for(FILTERS)]
    fake.hashes[cache.embeddings_key_for(FILTERS)][field] = json.dumps(
        {"embedding": EMBEDDING.tolist(), "filter_str": json.dumps(FILTERS, sort_keys=True)}
    ).encode()
    assert cache.get("a reworded question", EMBEDDING, FILTERS) == [{"content": "text"}]


def test_embedding_of_a_different_size_is_ignored(cache):
    cache.set("original question", EMBEDDING, FILTERS, [{"content": "text"}])
    assert cache.get("a reworded question", np.ones(5), FILTERS) is None


def test_unreachable_redis_gives_no_cache_rather_than_an_error(monkeypatch):
    from app.services import query_cache

    def unreachable():
        raise redis.ConnectionError("refused")

    monkeypatch.setattr(query_cache, "get_query_cache", unreachable)
    assert query_cache.get_query_cache_or_none() is None


def test_similarity_scan_checks_entries_in_one_round_trip(cache, fake):
    vectors = np.random.default_rng(0).normal(size=(50, 16))  # clearly distinct directions
    for i, vector in enumerate(vectors):
        cache.set(f"question {i}", vector, FILTERS, [{"content": str(i)}])
    # Five entries have expired in Redis: their embedding references are cleaned up
    for i in range(5):
        del fake.values[cache.cache_key_prefix + cache._create_cache_key(f"question {i}", FILTERS)]
    fake.round_trips = 0

    assert cache.get("a new question", vectors[49], FILTERS) == [{"content": "49"}]
    assert fake.round_trips == 1
    assert len(fake.hashes[cache.embeddings_key_for(FILTERS)]) == 45


def test_cache_leaves_redis_alone_for_a_while_after_a_failure(cache, fake):
    from app.services import query_cache

    cache.set("q", EMBEDDING, FILTERS, [{"content": "text"}])
    fake.failing = {"get"}
    assert cache.get("q", EMBEDDING, FILTERS) is None
    assert query_cache.redis_in_backoff()

    fake.failing = set()
    fake.calls = 0
    assert cache.get("q", EMBEDDING, FILTERS) is None  # not even tried
    cache.set("other", EMBEDDING, FILTERS, [{"content": "x"}])
    assert query_cache.get_query_cache_or_none() is None
    assert fake.calls == 0

    query_cache._redis_retry_at = 0.0  # the backoff period is over
    assert cache.get("q", EMBEDDING, FILTERS) == [{"content": "text"}]


def test_redis_client_fails_fast_without_retries(monkeypatch):
    from app.services import query_cache

    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def ping(self):
            return True

    monkeypatch.setattr(query_cache, "_redis_client", None)
    monkeypatch.setattr(query_cache.redis, "Redis", FakeClient)
    query_cache.get_redis_client()
    assert captured["socket_connect_timeout"] <= 2
    assert captured["retry"].get_retries() == 0


def test_legacy_pickled_entries_are_misses(cache, fake):
    import pickle

    key = cache.cache_key_prefix + cache._create_cache_key("q", FILTERS)
    fake.values[key] = pickle.dumps({"results": ["old"]})
    fake.hashes[cache.embeddings_key_for(FILTERS)] = {b"legacy": pickle.dumps({"embedding": [1.0]})}
    assert cache.get("q", EMBEDDING, FILTERS) is None
    assert b"legacy" not in fake.hashes[cache.embeddings_key_for(FILTERS)]


# --- one embeddings hash per filter set --------------------------------------------------------

OTHER_FILTERS = {"tab": "digital-id"}


def test_a_lookup_reads_only_the_embeddings_cached_with_the_same_filters(cache, fake):
    cache.set("q1", EMBEDDING, FILTERS, [{"content": "a"}])
    cache.set("q2", EMBEDDING, OTHER_FILTERS, [{"content": "b"}])
    assert cache.embeddings_key_for(FILTERS) != cache.embeddings_key_for(OTHER_FILTERS)
    assert len(fake.hashes[cache.embeddings_key_for(FILTERS)]) == 1

    read = []
    original = fake.hgetall
    fake.hgetall = lambda key: read.append(key) or original(key)
    assert cache.get("a reworded question", EMBEDDING, OTHER_FILTERS) == [{"content": "b"}]
    assert read == [cache.embeddings_key_for(OTHER_FILTERS)]


def test_each_filter_hash_expires_with_its_entries(cache, fake):
    cache.set("q", EMBEDDING, FILTERS, [{"content": "a"}])
    assert fake.expiries[cache.embeddings_key_for(FILTERS)] == cache.ttl


def test_a_filter_hash_is_trimmed_to_its_newest_entries(cache, fake, monkeypatch):
    from app.services import query_cache

    monkeypatch.setattr(query_cache, "MAX_EMBEDDINGS_PER_FILTER", 3)
    clock = iter(range(100, 200))
    monkeypatch.setattr(query_cache.time, "time", lambda: next(clock))
    for i in range(5):
        cache.set(f"question {i}", np.eye(5)[i], FILTERS, [{"content": str(i)}])
    kept = {_loads(v)["stored_at"] for v in fake.hashes[cache.embeddings_key_for(FILTERS)].values()}
    assert len(kept) == 3 and min(kept) > 101  # the two oldest were dropped


def test_the_old_single_embeddings_hash_is_removed(fake):
    fake.hashes["query_cache:embeddings"] = {b"old": b"{}"}
    QueryCache(redis_client=fake)
    assert "query_cache:embeddings" not in fake.hashes
