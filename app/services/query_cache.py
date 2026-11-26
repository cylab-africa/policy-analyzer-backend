"""
Query Cache Service for Embeddings

Implements semantic caching using Redis to avoid redundant embedding generation 
and vector searches for similar queries. Uses cosine similarity to match queries.
"""

from typing import List, Dict, Any, Optional, Tuple
import base64
import binascii
import numpy as np
from datetime import date, datetime, timedelta
from decimal import Decimal
import hashlib
import json
import threading
import time

import redis
from bson import ObjectId
from redis import Redis
from redis.backoff import NoBackoff
from redis.retry import Retry


# While Redis is down, every command would wait for a connection timeout. After a failure,
# callers (the query cache and the rate limiter) skip Redis for this long, then try again.
REDIS_RETRY_SECONDS = 30.0

_redis_retry_at = 0.0
_redis_state_lock = threading.Lock()


def redis_in_backoff() -> bool:
    """True for a while after a Redis failure: callers should not wait on Redis meanwhile."""
    return time.monotonic() < _redis_retry_at


def report_redis_failure(context: str, error: Exception) -> None:
    """
    Record a Redis failure. A lost connection or a timeout means Redis is unreachable, so it is
    left alone for REDIS_RETRY_SECONDS. Any other error (an error reply such as READONLY or OOM)
    came from a working server, so it is only logged.
    """
    global _redis_retry_at
    if not isinstance(error, (redis.ConnectionError, redis.TimeoutError)):
        print(f"[REDIS] Error during {context} (ignored): {error}")
        return
    with _redis_state_lock:
        first = not redis_in_backoff()
        _redis_retry_at = time.monotonic() + REDIS_RETRY_SECONDS
    if first:
        print(f"[REDIS] Unavailable during {context}; carrying on without it for "
              f"{REDIS_RETRY_SECONDS:.0f}s: {error}")


# Cache values are stored as JSON, never pickle: unpickling data from Redis can execute code,
# so anyone able to write to Redis would be able to run code in this process.
#
# Search results carry document metadata, which can hold values JSON has no type for (dates,
# decimals, MongoDB ids, sets). Those are written as {"__cache_type__": name, "value": ...} and
# turned back into the same type on read, so a cache hit returns what the original search did.
# A value of any other type makes _dumps raise TypeError; the caller then skips caching rather
# than storing something that would come back different.

_TYPE_TAG = "__cache_type__"

_DECODERS = {
    "datetime": datetime.fromisoformat,
    "date": date.fromisoformat,
    "decimal": Decimal,
    "objectid": ObjectId,
    "set": set,
}


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, datetime):  # checked before date: datetime is a subclass of date
        return {_TYPE_TAG: "datetime", "value": obj.isoformat()}
    if isinstance(obj, date):
        return {_TYPE_TAG: "date", "value": obj.isoformat()}
    if isinstance(obj, Decimal):
        return {_TYPE_TAG: "decimal", "value": str(obj)}
    if isinstance(obj, ObjectId):
        return {_TYPE_TAG: "objectid", "value": str(obj)}
    if isinstance(obj, (set, frozenset)):
        return {_TYPE_TAG: "set", "value": list(obj)}
    raise TypeError(f"{type(obj).__name__} values cannot be cached")


def _decode(obj: Dict[str, Any]) -> Any:
    decoder = _DECODERS.get(obj.get(_TYPE_TAG)) if len(obj) == 2 and "value" in obj else None
    if not decoder:
        return obj
    try:
        return decoder(obj["value"])
    except Exception as e:
        # Decimal raises InvalidOperation and ObjectId raises InvalidId, neither a ValueError.
        # Report every bad value the same way so callers treat the entry as unreadable.
        raise ValueError(f"Invalid cached {obj[_TYPE_TAG]} value") from e


def _dumps(value: Any) -> bytes:
    """Serialise a cache value to JSON bytes. Raises TypeError for a value it cannot restore."""
    return json.dumps(value, default=_encode).encode("utf-8")


def _loads(raw: Any) -> Any:
    """Parse a cache value. Raises ValueError for anything that is not a valid cache value."""
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8")  # a legacy pickled value fails here or below
    return json.loads(raw, object_hook=_decode)


# Embeddings are compared against every cached query on each lookup, so they are stored as raw
# float32 bytes (base64 inside the JSON record) instead of a JSON list of numbers.
#
# Only questions searched with the same filters can match, so embeddings are kept in one hash per
# filter set: a lookup reads the entries it could match, not every cached question. Each hash
# expires with the entries it points to, and holds at most MAX_EMBEDDINGS_PER_FILTER of them. Decoding that is
# a single copy, where parsing a 1536-number list costs far more. Still no pickle.

def _encode_embedding(embedding: Any) -> str:
    return base64.b64encode(np.asarray(embedding, dtype="<f4").tobytes()).decode("ascii")


def _decode_embedding(data: Dict[str, Any]) -> np.ndarray:
    if "embedding_f32" in data:
        try:
            raw = base64.b64decode(data["embedding_f32"], validate=True)
            return np.frombuffer(raw, dtype="<f4")
        except (binascii.Error, TypeError) as e:
            raise ValueError("Invalid cached embedding") from e
    # Entries written before the binary format
    return np.array(data["embedding"], dtype=float)


MAX_EMBEDDINGS_PER_FILTER = 500


class QueryCache:
    """
    Cache for query embeddings and their results using Redis.
    
    Prevents redundant embedding API calls by matching similar queries
    based on cosine similarity threshold.
    """
    
    def __init__(
        self,
        redis_client: Redis,
        similarity_threshold: float = 0.98,
        ttl_seconds: int = 3600,
        namespace: str = "query_cache"
    ):
        """
        Initialize query cache with Redis backend.
        
        Args:
            redis_client: Redis client instance
            similarity_threshold: Minimum cosine similarity to consider queries as matching (0.98 = 98%)
            ttl_seconds: Time-to-live for cache entries in seconds (default 1 hour)
            namespace: Redis key namespace prefix
        """
        self.redis = redis_client
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.namespace = namespace
        
        # Redis keys
        # One hash of embeddings per filter set: {namespace}:embeddings:<hash of the filters>
        self.embeddings_key_prefix = f"{namespace}:embeddings:"
        # The single hash used before, holding every cached question's embedding
        self.legacy_embeddings_key = f"{namespace}:embeddings"
        self.cache_key_prefix = f"{namespace}:entry:"    # Hash entries
        self.stats_key = f"{namespace}:stats"            # Stats hash

        # Initialize stats if not exist
        if not self.redis.exists(self.stats_key):
            self._reset_stats()
        # Its entries are not read any more; remove it rather than leave it to grow stale
        self._best_effort("delete", self.legacy_embeddings_key)

    def _embeddings_key(self, filter_str: str) -> str:
        return self.embeddings_key_prefix + hashlib.md5(filter_str.encode(), usedforsecurity=False).hexdigest()

    def embeddings_key_for(self, filters: Dict[str, Any]) -> str:
        """The Redis hash holding the embeddings of questions cached with these filters."""
        return self._embeddings_key(json.dumps(filters, sort_keys=True))

    def _best_effort(self, operation: str, *args) -> None:
        """
        Run a Redis write that only keeps the cache tidy (statistics, access times, removing
        stale entries). A failure is logged and ignored, so it never turns a cache hit into an
        error or stops a search from returning its results. Skipped while Redis is known to be
        down.
        """
        if redis_in_backoff():
            return
        try:
            getattr(self.redis, operation)(*args)
        except redis.RedisError as e:
            report_redis_failure(f"query cache {operation}", e)

    def _reset_stats(self):
        """Initialize or reset statistics."""
        self.redis.hset(self.stats_key, mapping={
            "hits": 0,
            "misses": 0,
            "total_queries": 0,
            "cache_savings_count": 0
        })
    
    def _create_cache_key(self, query: str, filters: Dict[str, Any]) -> str:
        """Create a unique cache key from query and filters."""
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Sort filters for consistent hashing
        filter_str = json.dumps(filters, sort_keys=True)
        
        # Create hash
        key_string = f"{normalized_query}||{filter_str}"
        return hashlib.md5(key_string.encode(), usedforsecurity=False).hexdigest()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _find_similar_query(
        self,
        query_embedding: np.ndarray,
        filters: Dict[str, Any]
    ) -> Optional[Tuple[str, float]]:
        """
        Find a cached query similar to the current one.
        
        Args:
            query_embedding: Embedding vector of the current query
            filters: Metadata filters for the query
            
        Returns:
            Tuple of (cache_key, similarity_score) if found, None otherwise
        """
        # Normalize filter for comparison
        filter_str = json.dumps(filters, sort_keys=True)
        embeddings_key = self._embeddings_key(filter_str)

        # The embeddings cached with these filters (stored as JSON in Redis)
        cached_entries = self.redis.hgetall(embeddings_key)

        if not cached_entries:
            return None

        unreadable: List[str] = []
        candidates: List[Tuple[float, str]] = []  # (similarity, cache_key), filters matching

        for cache_key_bytes, embedding_data in cached_entries.items():
            cache_key = cache_key_bytes.decode('utf-8')

            # Load metadata, and the embedding only when the filters match (must be exact)
            try:
                data = _loads(embedding_data)
                if data['filter_str'] != filter_str:
                    continue
                cached_embedding = _decode_embedding(data)
            except (ValueError, KeyError, TypeError):
                # Not valid JSON: an entry written by an older version (pickle) or a corrupt one.
                # Drop it so it is not re-read on every lookup.
                unreadable.append(cache_key)
                continue

            if cached_embedding.shape != np.shape(query_embedding):
                continue  # written with a different embedding model

            similarity = float(self._cosine_similarity(query_embedding, cached_embedding))
            candidates.append((similarity, cache_key))

        # Check which entries still exist (Redis expires them by TTL) in one round trip, rather
        # than one EXISTS per cached query. Stale embedding references are removed in passing.
        stale = []
        best = None
        if candidates:
            pipe = self.redis.pipeline(transaction=False)
            for _, cache_key in candidates:
                pipe.exists(f"{self.cache_key_prefix}{cache_key}")
            for (similarity, cache_key), exists in zip(candidates, pipe.execute()):
                if not exists:
                    stale.append(cache_key)
                elif best is None or similarity > best[0]:
                    best = (similarity, cache_key)

        if unreadable or stale:
            self._best_effort("hdel", embeddings_key, *(unreadable + stale))

        # Return match if above threshold
        if best and best[0] >= self.similarity_threshold:
            return (best[1], best[0])

        return None
    
    def get(
        self,
        query: str,
        query_embedding: np.ndarray,
        filters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached results for a query.
        
        Args:
            query: The query text
            query_embedding: The query embedding vector
            filters: Metadata filters applied to the query
            
        Returns:
            Cached results dict if found, None otherwise. A Redis failure counts as a miss, so
            the caller searches as usual instead of failing.
        """
        if redis_in_backoff():
            return None
        try:
            return self._get(query, query_embedding, filters)
        except redis.RedisError as e:
            report_redis_failure("query cache lookup", e)
            return None

    def _get(
        self,
        query: str,
        query_embedding: np.ndarray,
        filters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        # Increment total queries
        self._best_effort("hincrby", self.stats_key, "total_queries", 1)
        if redis_in_backoff():
            return None  # that write just failed: do not wait on Redis again for this search

        # Try exact match first
        cache_key = self._create_cache_key(query, filters)
        entry_key = f"{self.cache_key_prefix}{cache_key}"
        
        cached_data = self.redis.get(entry_key)
        if cached_data:
            try:
                entry = _loads(cached_data)
                results = entry['results']
            except (ValueError, KeyError, TypeError):
                pass  # unreadable or legacy entry: treat as a miss
            else:
                print(f"[QUERY CACHE] Exact match found for query: '{query[:50]}...'")
                self._record_hit(entry_key, entry)
                return results

        # Try semantic similarity match
        similar_match = self._find_similar_query(query_embedding, filters)

        if similar_match:
            matched_key, similarity = similar_match
            entry_key = f"{self.cache_key_prefix}{matched_key}"

            cached_data = self.redis.get(entry_key)
            if cached_data:
                try:
                    entry = _loads(cached_data)
                    results = entry['results']
                except (ValueError, KeyError, TypeError):
                    pass  # unreadable or legacy entry: treat as a miss
                else:
                    print(f"[QUERY CACHE] Similar query found (similarity: {similarity:.4f})")
                    print(f"  Original: '{str(entry.get('query', ''))[:60]}...'")
                    print(f"  Current:  '{query[:60]}...'")
                    self._best_effort("hincrby", self.stats_key, "cache_savings_count", 1)
                    self._record_hit(entry_key, entry)
                    return results

        # Cache miss
        self._best_effort("hincrby", self.stats_key, "misses", 1)
        return None

    def _record_hit(self, entry_key: str, entry: Dict[str, Any]) -> None:
        """Count a hit and refresh the entry's access time and TTL. Never raises."""
        self._best_effort("hincrby", self.stats_key, "hits", 1)
        entry['last_accessed'] = datetime.now().isoformat()
        entry['access_count'] = entry.get('access_count', 0) + 1
        try:
            refreshed = _dumps(entry)
        except (TypeError, ValueError):
            return
        self._best_effort("setex", entry_key, self.ttl, refreshed)
    
    def set(
        self,
        query: str,
        query_embedding: np.ndarray,
        filters: Dict[str, Any],
        results: List[Dict[str, Any]]
    ):
        """
        Cache query results in Redis.

        Caching is an optimisation, so this never raises: results that cannot be stored
        faithfully, or a Redis failure, just mean the query is not cached.

        Args:
            query: The query text
            query_embedding: The query embedding vector
            filters: Metadata filters applied to the query
            results: Search results to cache
        """
        cache_key = self._create_cache_key(query, filters)
        entry_key = f"{self.cache_key_prefix}{cache_key}"

        # Store cache entry with TTL
        entry = {
            "query": query,
            "filters": filters,
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "access_count": 0
        }

        filter_str = json.dumps(filters, sort_keys=True)
        embeddings_key = self._embeddings_key(filter_str)
        try:
            entry_data = _dumps(entry)
            embedding_data = _dumps({
                'embedding_f32': _encode_embedding(query_embedding),
                'filter_str': filter_str,
                'stored_at': time.time(),
            })
        except (TypeError, ValueError) as e:
            print(f"[QUERY CACHE] Not caching query (results cannot be stored): {e}")
            return

        if redis_in_backoff():
            return
        try:
            self.redis.setex(entry_key, self.ttl, entry_data)
            # Store embedding for similarity matching
            self.redis.hset(embeddings_key, cache_key, embedding_data)
        except redis.RedisError as e:
            report_redis_failure("query cache store", e)
            return

        # The hash goes once none of its entries can still exist
        self._best_effort("expire", embeddings_key, self.ttl)
        self._limit_embeddings(embeddings_key)
        print(f"[QUERY CACHE] Cached new query: '{query[:50]}...'")

    def _limit_embeddings(self, embeddings_key: str) -> None:
        """
        Keep a filter's hash within MAX_EMBEDDINGS_PER_FILTER: drop references to entries Redis
        has expired, then the oldest. Best effort, like the other bookkeeping.
        """
        if redis_in_backoff():
            return
        try:
            if self.redis.hlen(embeddings_key) <= MAX_EMBEDDINGS_PER_FILTER:
                return
            records = self.redis.hgetall(embeddings_key)
            fields = [f.decode("utf-8") if isinstance(f, bytes) else f for f in records]
            pipe = self.redis.pipeline(transaction=False)
            for field in fields:
                pipe.exists(f"{self.cache_key_prefix}{field}")
            alive = []
            stale = []
            for field, exists in zip(fields, pipe.execute()):
                if not exists:
                    stale.append(field)
                    continue
                try:
                    stored_at = float(_loads(records[field.encode("utf-8")]).get("stored_at", 0))
                except (ValueError, KeyError, TypeError):
                    stored_at = 0.0  # unreadable: drop it first
                alive.append((stored_at, field))
            alive.sort()
            excess = [field for _, field in alive[:max(0, len(alive) - MAX_EMBEDDINGS_PER_FILTER)]]
            if stale or excess:
                self.redis.hdel(embeddings_key, *(stale + excess))
        except redis.RedisError as e:
            report_redis_failure("query cache trim", e)
    
    def clear(self):
        """Clear all cache entries."""
        # Get all entry keys
        pattern = f"{self.cache_key_prefix}*"
        cursor = 0
        
        while True:
            cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
            if keys:
                self.redis.delete(*keys)
            if cursor == 0:
                break
        
        # Clear embeddings (every filter's hash, and the old single hash)
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=f"{self.embeddings_key_prefix}*", count=100)
            if keys:
                self.redis.delete(*keys)
            if cursor == 0:
                break
        self.redis.delete(self.legacy_embeddings_key)
        
        # Reset stats
        self._reset_stats()
        
        print("[QUERY CACHE] Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats_data = self.redis.hgetall(self.stats_key)
        
        stats = {
            "hits": int(stats_data.get(b"hits", 0)),
            "misses": int(stats_data.get(b"misses", 0)),
            "total_queries": int(stats_data.get(b"total_queries", 0)),
            "cache_savings_count": int(stats_data.get(b"cache_savings_count", 0))
        }
        
        total = stats["hits"] + stats["misses"]
        hit_rate = (stats["hits"] / total * 100) if total > 0 else 0
        
        # Get cache size: embeddings across every filter's hash
        cache_size = sum(
            self.redis.hlen(key)
            for key in self.redis.scan_iter(match=f"{self.embeddings_key_prefix}*", count=100)
        )
        
        return {
            **stats,
            "hit_rate_percentage": round(hit_rate, 2),
            "cache_size": cache_size,
            "estimated_api_calls_saved": stats["cache_savings_count"]
        }
    
    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()
        
        print("\n" + "="*50)
        print("QUERY CACHE STATISTICS (Redis)")
        print("="*50)
        print(f"Total Queries:           {stats['total_queries']}")
        print(f"Cache Hits:              {stats['hits']}")
        print(f"Cache Misses:            {stats['misses']}")
        print(f"Hit Rate:                {stats['hit_rate_percentage']}%")
        print(f"Cache Size:              {stats['cache_size']}")
        print(f"API Calls Saved:         {stats['estimated_api_calls_saved']}")
        print(f"Similarity Threshold:    {self.similarity_threshold}")
        print(f"TTL:                     {self.ttl}s")
        print("="*50 + "\n")


# Global cache instance
_query_cache: Optional[QueryCache] = None
_redis_client: Optional[Redis] = None


def get_redis_client() -> Redis:
    """Get or create Redis client."""
    global _redis_client
    
    if _redis_client is None:
        from app.core.config import settings
        
        _redis_client = redis.Redis(
            host=getattr(settings, 'redis_host', 'localhost'),
            port=getattr(settings, 'redis_port', 6379),
            db=getattr(settings, 'redis_db', 0),
            password=getattr(settings, 'redis_password', None),
            decode_responses=False,  # Values are read as bytes and decoded in _loads
            # Fail fast when Redis is unreachable: callers then carry on without the cache (and
            # the rate limiter counts in memory) instead of hanging the request. redis-py retries
            # each command with backoff by default, which would multiply these timeouts, so a
            # failed command is not retried; the backoff below decides when to try Redis again.
            socket_connect_timeout=2,
            socket_timeout=5,
            retry=Retry(NoBackoff(), 0),
        )
        
        # Test connection
        try:
            _redis_client.ping()
            print(f"[REDIS] Connected to Redis at {getattr(settings, 'redis_host', 'localhost')}:{getattr(settings, 'redis_port', 6379)}")
        except redis.ConnectionError as e:
            print(f"[REDIS] Failed to connect to Redis: {e}")
            raise
    
    return _redis_client


def get_query_cache(
    similarity_threshold: float = 0.98,
    ttl_seconds: int = 3600
) -> QueryCache:
    """
    Get or create the global query cache instance with Redis backend.
    
    Args:
        similarity_threshold: Minimum similarity to consider queries as matching
        ttl_seconds: Time-to-live for cache entries in seconds (default 1 hour)
        
    Returns:
        QueryCache instance with Redis backend
    """
    global _query_cache
    
    if _query_cache is None:
        redis_client = get_redis_client()
        
        _query_cache = QueryCache(
            redis_client=redis_client,
            similarity_threshold=similarity_threshold,
            ttl_seconds=ttl_seconds
        )
        print(f"[QUERY CACHE] Initialized with Redis backend (threshold={similarity_threshold}, ttl={ttl_seconds}s)")

    return _query_cache


def get_query_cache_or_none() -> Optional[QueryCache]:
    """
    The query cache, or None while Redis cannot be reached.

    Searches use this so that a Redis outage only costs the cache, not the search results.
    """
    if redis_in_backoff():
        return None
    try:
        return get_query_cache()
    except redis.RedisError as e:
        report_redis_failure("query cache start-up", e)
        return None

