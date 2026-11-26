"""Checks for access control.

They need no database, API keys or network: the pieces are tested on their own or mounted on
a tiny FastAPI app.
"""
import os

import pytest
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app.core.config import Settings, settings
from app.core.job_tokens import hash_token, new_token, token_matches
from app.core.safe_paths import resolve_within
from app.middleware.auth import APIKeyMiddleware
import redis

from app.middleware.rate_limit import RedisWindowLimiter, SlidingWindowLimiter, client_id, rate_limit


# --- delete tokens ------------------------------------------------------------------------

def test_token_matches_only_its_own_hash():
    token = new_token()
    stored = hash_token(token)
    assert token_matches(token, stored)
    assert not token_matches(new_token(), stored)


@pytest.mark.parametrize("token,stored", [(None, "abc"), ("abc", None), ("", "abc"), ("abc", "")])
def test_missing_token_or_hash_never_matches(token, stored):
    assert not token_matches(token, stored)


def test_tokens_are_unique_and_long():
    assert len({new_token() for _ in range(50)}) == 50
    assert len(new_token()) >= 40


# --- folder-process path allow-list --------------------------------------------------------

def test_resolve_within_accepts_paths_inside_root(tmp_path):
    (tmp_path / "docs").mkdir()
    assert resolve_within(tmp_path, tmp_path / "docs") == (tmp_path / "docs").resolve()
    assert resolve_within(tmp_path, "docs") == (tmp_path / "docs").resolve()  # relative to root


def test_resolve_within_rejects_traversal_and_outside_paths(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    assert resolve_within(root, "../outside") is None
    assert resolve_within(root, outside) is None
    assert resolve_within(root, root / "missing") is None
    assert resolve_within(tmp_path / "no-such-root", ".") is None


def test_resolve_within_rejects_symlink_escaping_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("x")
    link = root / "link.txt"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available here")
    assert resolve_within(root, link) is None


# --- rate limiting -------------------------------------------------------------------------

def test_limiter_blocks_after_limit_and_recovers():
    now = [0.0]
    limiter = SlidingWindowLimiter(limit=2, window=60, clock=lambda: now[0])
    assert limiter.check("a")[0] and limiter.check("a")[0]
    allowed, retry_after = limiter.check("a")
    assert not allowed and 0 < retry_after <= 60
    assert limiter.check("b")[0]  # another client is unaffected
    now[0] = 61
    assert limiter.check("a")[0]


def test_limit_of_zero_switches_the_limiter_off():
    limiter = SlidingWindowLimiter(limit=0, window=60)
    assert all(limiter.check("a")[0] for _ in range(100))


def _limited_app(limit=2):
    app = FastAPI()
    guard = rate_limit("test", lambda: limit, window=60)

    @app.get("/ping", dependencies=[Depends(guard)])
    def ping():
        return {"ok": True}

    return TestClient(app)


def test_rate_limit_dependency_returns_429_with_retry_after():
    client = _limited_app(limit=2)
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200
    blocked = client.get("/ping")
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1


def _request(peer, client_ip):
    return type("Req", (), {
        "client": type("C", (), {"host": peer})(),
        "headers": {"x-client-ip": client_ip},
    })()


def test_client_ip_header_is_only_believed_from_a_trusted_proxy(monkeypatch):
    monkeypatch.setattr(settings, "trusted_proxy_ips", "")
    assert client_id(_request("10.0.0.5", "203.0.113.9")) == "10.0.0.5"

    monkeypatch.setattr(settings, "trusted_proxy_ips", "10.0.0.5, 10.0.0.9/32")
    assert client_id(_request("10.0.0.5", "203.0.113.9")) == "203.0.113.9"
    assert client_id(_request("10.0.0.9", "203.0.113.9")) == "203.0.113.9"  # /32 = one address
    # A direct caller cannot forge the header to dodge its limit or use up someone else's
    assert client_id(_request("198.51.100.7", "203.0.113.9")) == "198.51.100.7"
    # A malformed header falls back to the peer address
    assert client_id(_request("10.0.0.5", "not-an-ip")) == "10.0.0.5"


def test_address_ranges_are_refused_because_they_include_the_gateway(monkeypatch, capsys):
    monkeypatch.setattr(settings, "trusted_proxy_ips", "172.18.0.0/16")
    # Docker forwards some outside connections from the network's gateway
    assert client_id(_request("172.18.0.1", "203.0.113.9")) == "172.18.0.1"
    assert client_id(_request("172.18.4.2", "203.0.113.9")) == "172.18.4.2"
    assert "not ranges" in capsys.readouterr().out


def test_invalid_trusted_proxy_entries_are_ignored(monkeypatch):
    monkeypatch.setattr(settings, "trusted_proxy_ips", "not a host!, 10.0.0.5")
    assert client_id(_request("10.0.0.5", "203.0.113.9")) == "203.0.113.9"
    assert client_id(_request("10.0.0.6", "203.0.113.9")) == "10.0.0.6"


def test_trusted_proxy_can_be_named_by_hostname(monkeypatch):
    from app.middleware import rate_limit as rl

    addresses = {"pdp-frontend": "172.18.0.7"}

    def fake_getaddrinfo(host, port):
        if host not in addresses:
            raise OSError("unknown host")
        return [(None, None, None, None, (addresses[host], 0))]

    monkeypatch.setattr(rl.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(rl, "_resolved", {})
    monkeypatch.setattr(settings, "trusted_proxy_ips", "pdp-frontend")
    assert client_id(_request("172.18.0.7", "203.0.113.9")) == "203.0.113.9"
    assert client_id(_request("172.18.0.1", "203.0.113.9")) == "172.18.0.1"

    # The container is recreated with a new address: picked up once the lookup expires
    addresses["pdp-frontend"] = "172.18.0.8"
    monkeypatch.setattr(rl, "_resolved", {})
    assert client_id(_request("172.18.0.8", "203.0.113.9")) == "203.0.113.9"


def test_ignored_client_ip_header_is_reported_once(monkeypatch, capsys):
    from app.middleware import rate_limit as rl

    monkeypatch.setattr(rl, "_warned_untrusted", set())
    monkeypatch.setattr(settings, "trusted_proxy_ips", "")
    client_id(_request("172.18.0.7", "203.0.113.9"))
    client_id(_request("172.18.0.7", "203.0.113.10"))
    out = capsys.readouterr().out
    assert out.count("Ignoring X-Client-IP from 172.18.0.7") == 1
    assert "TRUSTED_PROXY_IPS" in out


def test_ipv6_clients_are_counted_per_64(monkeypatch):
    monkeypatch.setattr(settings, "trusted_proxy_ips", "10.0.0.5")
    one = client_id(_request("2001:db8:1:2:aaaa::1", None))
    other_address_same_64 = client_id(_request("2001:db8:1:2:ffff:ffff:ffff:ffff", None))
    next_64 = client_id(_request("2001:db8:1:3::1", None))
    assert one == other_address_same_64 == "2001:db8:1:2::/64"
    assert next_64 != one
    # Through the website's server too
    assert client_id(_request("10.0.0.5", "2001:db8:1:2::99")) == "2001:db8:1:2::/64"
    # An IPv4 client seen on an IPv6 socket is still counted by its IPv4 address
    assert client_id(_request("::ffff:198.51.100.7", None)) == "198.51.100.7"


class FakePipeline:
    def __init__(self, client):
        self.client, self.ops = client, []

    def incr(self, key):
        self.ops.append(("incr", key))

    def expire(self, key, seconds):
        self.ops.append(("expire", key))

    def get(self, key):
        self.ops.append(("get", key))

    def execute(self):
        if self.client.fail:
            raise redis.ConnectionError("down")
        results = []
        for op, key in self.ops:
            if op == "incr":
                self.client.store[key] = self.client.store.get(key, 0) + 1
                results.append(self.client.store[key])
            elif op == "get":
                value = self.client.store.get(key)
                results.append(None if value is None else str(value).encode())
            else:
                results.append(True)
        return results


class FakeRedisClient:
    def __init__(self):
        self.store, self.fail = {}, False

    def pipeline(self, transaction=True):
        return FakePipeline(self)

    def decr(self, key):
        self.store[key] -= 1


def test_redis_limiter_shares_counts_between_limiters():
    shared = FakeRedisClient()
    now = [1000.0]
    worker_a = RedisWindowLimiter("q", 2, 60, lambda: shared, clock=lambda: now[0])
    worker_b = RedisWindowLimiter("q", 2, 60, lambda: shared, clock=lambda: now[0])
    assert worker_a.check("1.2.3.4")[0]
    assert worker_b.check("1.2.3.4")[0]
    allowed, retry_after = worker_a.check("1.2.3.4")  # third hit, from either worker
    assert not allowed
    assert 0 < retry_after <= 120
    assert worker_b.check("5.6.7.8")[0]  # another client is unaffected
    now[0] += 120  # both earlier windows have slid out
    assert worker_a.check("1.2.3.4")[0]


def test_redis_limiter_blocks_a_burst_across_the_window_boundary():
    shared = FakeRedisClient()
    now = [1019.9]  # 0.1 s before a 60 s window ends (windows start at multiples of 60)
    limiter = RedisWindowLimiter("q", 20, 60, lambda: shared, clock=lambda: now[0])
    assert all(limiter.check("c")[0] for _ in range(20))
    now[0] = 1020.0  # a new fixed window, but the sliding window still holds those 20
    allowed, retry_after = limiter.check("c")
    assert not allowed
    assert retry_after > 0
    # Rejected requests do not use up the limit
    assert sum(shared.store.values()) == 20
    now[0] = 1020.0 + 60
    assert limiter.check("c")[0]


def test_redis_limiter_lets_requests_through_as_the_old_window_slides_out():
    shared = FakeRedisClient()
    now = [960.0]
    limiter = RedisWindowLimiter("q", 10, 60, lambda: shared, clock=lambda: now[0])
    assert all(limiter.check("c")[0] for _ in range(10))
    now[0] = 1020.0 + 30  # half of the previous window still overlaps: it counts as 5
    assert sum(limiter.check("c")[0] for _ in range(10)) == 5


def test_rate_limit_uses_redis_and_falls_back_to_memory(monkeypatch):
    from app.middleware import rate_limit as rl

    fake = FakeRedisClient()
    monkeypatch.setattr(rl, "_redis_client", lambda: fake)
    monkeypatch.setattr(settings, "rate_limit_store", "redis")
    guard = rate_limit("test", lambda: 1, window=60)
    app = FastAPI()

    @app.get("/ping", dependencies=[Depends(guard)])
    def ping():
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 429
    assert sum(fake.store.values()) == 1  # counted in Redis; the rejected request was not

    fake.fail = True  # Redis goes down: the in-memory limiter takes over, still limiting
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 429

    # Redis is not retried on every request while it is down
    calls = []
    monkeypatch.setattr(rl, "_redis_client", lambda: calls.append(1) or fake)
    client.get("/ping")
    assert calls == []


def test_limiter_reads_a_changed_limit_and_can_be_reset():
    limit = [1]
    guard = rate_limit("test", lambda: limit[0], window=60)
    app = FastAPI()

    @app.get("/ping", dependencies=[Depends(guard)])
    def ping():
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 429
    limit[0] = 3  # a new setting applies without rebuilding the dependency
    assert client.get("/ping").status_code == 200
    guard.limiter.reset()
    assert all(client.get("/ping").status_code == 200 for _ in range(3))
    assert client.get("/ping").status_code == 429


# --- API key check -------------------------------------------------------------------------

def _key_client():
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware, excluded_paths=["/"])

    @app.get("/")
    def root():
        return {"ok": True}

    @app.get("/secret")
    def secret():
        return {"secret": True}

    return TestClient(app)


def test_api_key_required_and_checked():
    client = _key_client()
    assert client.get("/").status_code == 200
    assert client.get("/secret").status_code == 401
    assert client.get("/secret", headers={"X-API-Key": "wrong"}).status_code == 403
    assert client.get("/secret", headers={"X-API-Key": settings.api_key}).status_code == 200
    assert client.get("/secret", headers={"Authorization": f"Bearer {settings.api_key}"}).status_code == 200


def test_non_ascii_key_is_rejected_not_a_server_error():
    client = _key_client()
    assert client.get("/secret", headers={"X-API-Key": "café".encode("latin-1")}).status_code == 403


# --- CORS ----------------------------------------------------------------------------------

def test_wildcard_is_never_an_allowed_origin():
    s = Settings(openai_api_key="k", mongodb_connection_string="m", api_key="a",
                 cors_allowed_origins="*, https://app.example.org/ ,")
    assert s.cors_origins == ["https://app.example.org"]
    assert Settings(openai_api_key="k", mongodb_connection_string="m", api_key="a").cors_origins == []


def test_cors_headers_only_for_listed_origins():
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["https://app.example.org"], allow_credentials=False,
                       allow_methods=["GET"], allow_headers=["X-API-Key"])

    @app.get("/x")
    def x():
        return {}

    client = TestClient(app)
    ok = client.get("/x", headers={"Origin": "https://app.example.org"})
    assert ok.headers.get("access-control-allow-origin") == "https://app.example.org"
    bad = client.get("/x", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers
