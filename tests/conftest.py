"""Test setup: give the settings object the values it requires, without touching real services."""
import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017")
os.environ.setdefault("API_KEY", "test-api-key")
# Count rate limits in memory so tests never try to reach a Redis server
os.environ.setdefault("RATE_LIMIT_STORE", "memory")


@pytest.fixture(autouse=True)
def _redis_considered_up(monkeypatch):
    """A simulated Redis outage in one test must not make Redis look down in the next."""
    from app.services import query_cache

    monkeypatch.setattr(query_cache, "_redis_retry_at", 0.0)
