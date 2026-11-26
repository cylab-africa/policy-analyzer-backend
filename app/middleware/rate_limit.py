"""
Rate limiting for the endpoints that cost money or heavy processing.

Counts are kept in Redis (the same server as the query cache), so the limit holds across uvicorn
workers, replicas and restarts. While Redis cannot be reached, each process falls back to its own
in-memory counter, so a limit still applies, just per process. Both count over a sliding window,
so a client cannot fit twice the limit around a window boundary.
"""
import ipaddress
import re
import socket
import threading
import time
from collections import deque
from functools import lru_cache
from typing import Callable, Deque, Dict, FrozenSet, Optional, Tuple, Union

import redis
from fastapi import HTTPException, Request, status

from app.core.config import settings

Limit = Union[int, Callable[[], int]]

# An IPv6 user normally controls a whole /64, so counting each address separately would let them
# rotate addresses to get a fresh limit for every request.
IPV6_PREFIX = 64

# How long a hostname in TRUSTED_PROXY_IPS stays resolved before it is looked up again. Short,
# so a recreated website container (which gets a new address) is recognised again quickly.
HOSTNAME_TTL_SECONDS = 30.0


def _read_limit(limit: Limit) -> int:
    return int(limit() if callable(limit) else limit)


class SlidingWindowLimiter:
    """
    In-memory limiter: at most ``limit`` hits per ``window`` seconds for each key, in this process.

    ``limit`` may be a number or a callable. A callable is read on every check, so a changed
    setting takes effect straight away.
    """

    def __init__(self, limit: Limit, window: float, clock: Callable[[], float] = time.monotonic):
        self._limit = limit
        self.window = window
        self._clock = clock
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return _read_limit(self._limit)

    def check(self, key: str) -> Tuple[bool, float]:
        """Record a hit. Returns (allowed, seconds_until_next_allowed)."""
        limit = self.limit
        if limit <= 0:  # a limit of 0 switches the limiter off
            return True, 0.0

        now = self._clock()
        cutoff = now - self.window
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= limit:
                return False, max(hits[0] + self.window - now, 0.0)

            hits.append(now)
            if len(self._hits) > 10_000:
                self._prune(cutoff)
            return True, 0.0

    def reset(self) -> None:
        """Forget every recorded hit."""
        with self._lock:
            self._hits.clear()

    def _prune(self, cutoff: float) -> None:
        """Forget keys with no recent hits so the table cannot grow without bound."""
        for key in [k for k, h in self._hits.items() if not h or h[-1] <= cutoff]:
            del self._hits[key]


class RedisWindowLimiter:
    """
    Shared limiter counted in Redis, at most ``limit`` hits per sliding ``window`` for each key.

    It keeps one counter per fixed window and weights the previous window's count by how much of
    it still overlaps the sliding window (the standard sliding-window-counter method). That stops
    the burst a plain fixed window allows: 20 requests at 00:59 and 20 more at 01:00 are no longer
    both accepted. Rejected requests are not counted. Raises redis.RedisError when Redis cannot
    be used.
    """

    def __init__(
        self,
        name: str,
        limit: Limit,
        window: float,
        client: Callable[[], "redis.Redis"],
        clock: Callable[[], float] = time.time,
    ):
        self.name = name
        self._limit = limit
        self.window = window
        self._client = client
        self._clock = clock

    def _key(self, window_index: int, key: str) -> str:
        return f"rate_limit:{self.name}:{window_index}:{key}"

    def check(self, key: str) -> Tuple[bool, float]:
        limit = _read_limit(self._limit)
        if limit <= 0:
            return True, 0.0

        now = self._clock()
        index = int(now // self.window)
        elapsed = now - index * self.window
        current_key = self._key(index, key)

        client = self._client()
        pipe = client.pipeline(transaction=True)
        pipe.incr(current_key)
        # Kept for two windows: the next window still reads it as its "previous" count
        pipe.expire(current_key, int(self.window * 2) + 1)
        pipe.get(self._key(index - 1, key))
        count, _, previous = pipe.execute()
        count, previous = int(count), int(previous or 0)

        overlap = (self.window - elapsed) / self.window
        if previous * overlap + count <= limit:
            return True, 0.0

        client.decr(current_key)  # a rejected request does not use up the limit
        return False, self._retry_after(limit, previous, count - 1, elapsed)

    def _retry_after(self, limit: int, previous: int, current: int, elapsed: float) -> float:
        """Seconds until one more request would fit, given the counts before this request."""
        remaining = self.window - elapsed
        if current + 1 <= limit and previous > 0:
            # Fits once enough of the previous window has slid out
            wait = remaining - self.window * (limit - current - 1) / previous
        else:
            # Only in the next window, once enough of this window has slid out
            fraction = 1 - (limit - 1) / current if current > 0 else 0
            wait = remaining + self.window * max(fraction, 0)
        return max(wait, 0.0)


# --- who is the caller ---------------------------------------------------------------------

_HOSTNAME = re.compile(r"^(?=.{1,253}$)[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$")


@lru_cache(maxsize=8)
def _parse_trusted(value: str) -> Tuple[FrozenSet[ipaddress._BaseAddress], Tuple[str, ...]]:
    """
    Split TRUSTED_PROXY_IPS into exact addresses and hostnames.

    Address ranges are refused: a Docker network range includes its gateway, which is where Docker
    forwards some outside connections from, so trusting a range would let anyone forge the
    X-Client-IP header. A single-address range such as 10.0.0.5/32 is accepted as that address.
    """
    addresses, hostnames = set(), []
    for item in (part.strip() for part in value.split(",")):
        if not item:
            continue
        address = _parse_ip(item)
        if address is not None:
            addresses.add(address)
            continue
        try:
            network = ipaddress.ip_network(item, strict=False)
        except ValueError:
            network = None
        if network is not None:
            if network.num_addresses == 1:
                addresses.add(_parse_ip(str(network.network_address)))
            else:
                print(f"[RATE LIMIT] Ignoring TRUSTED_PROXY_IPS entry {item!r}: list exact addresses "
                      f"or hostnames, not ranges (a range includes the network's gateway)")
            continue
        if _HOSTNAME.match(item):
            hostnames.append(item.lower())
        else:
            print(f"[RATE LIMIT] Ignoring invalid TRUSTED_PROXY_IPS entry: {item!r}")
    return frozenset(addresses), tuple(hostnames)


_resolved: Dict[str, Tuple[float, FrozenSet[ipaddress._BaseAddress]]] = {}
_resolved_lock = threading.Lock()


def _resolve(hostname: str) -> FrozenSet[ipaddress._BaseAddress]:
    """Addresses of ``hostname``, looked up again every HOSTNAME_TTL_SECONDS."""
    now = time.monotonic()
    with _resolved_lock:
        cached = _resolved.get(hostname)
        if cached and cached[0] > now:
            return cached[1]
    try:
        infos = socket.getaddrinfo(hostname, None)
        found = frozenset(a for a in (_parse_ip(info[4][0]) for info in infos) if a is not None)
    except OSError as e:
        print(f"[RATE LIMIT] Could not resolve TRUSTED_PROXY_IPS hostname {hostname!r}: {e}")
        found = frozenset()
    with _resolved_lock:
        _resolved[hostname] = (now + HOSTNAME_TTL_SECONDS, found)
    return found


def _parse_ip(value: Optional[str]) -> Optional[ipaddress._BaseAddress]:
    try:
        address = ipaddress.ip_address((value or "").strip())
    except ValueError:
        return None
    # An IPv4 client seen through an IPv6 socket (::ffff:1.2.3.4) is still that IPv4 client
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        return address.ipv4_mapped
    return address


def _bucket(address: ipaddress._BaseAddress) -> str:
    """The rate-limit key for an address: the address itself, or its /64 for IPv6."""
    if isinstance(address, ipaddress.IPv6Address):
        return str(ipaddress.ip_network(f"{address}/{IPV6_PREFIX}", strict=False))
    return str(address)


def _is_trusted_proxy(address: Optional[ipaddress._BaseAddress]) -> bool:
    if address is None:
        return False
    addresses, hostnames = _parse_trusted(settings.trusted_proxy_ips or "")
    if address in addresses:
        return True
    return any(address in _resolve(name) for name in hostnames)


_warned_untrusted: set = set()


def _warn_ignored_header(peer: str) -> None:
    """Say once per address that X-Client-IP is being ignored, instead of failing silently."""
    if peer in _warned_untrusted or len(_warned_untrusted) > 100:
        return
    _warned_untrusted.add(peer)
    print(f"[RATE LIMIT] Ignoring X-Client-IP from {peer}: that address is not in TRUSTED_PROXY_IPS, "
          f"so all of its requests share one rate limit. If {peer} is the website's server, add it "
          f"(or its hostname) to TRUSTED_PROXY_IPS.")


def client_id(request: Request) -> str:
    """
    Identify the caller for rate limiting.

    The website's server sits in front of this API and puts the visitor's address in
    ``X-Client-IP``. That header is believed only when the request itself comes from an address
    listed (or a hostname named) in TRUSTED_PROXY_IPS, so a caller that reaches the API directly
    cannot forge it. Everyone else is identified by the address the connection comes from.
    """
    peer_text = request.client.host if request.client else None
    peer = _parse_ip(peer_text)
    header = request.headers.get("x-client-ip")
    if _is_trusted_proxy(peer):
        visitor = _parse_ip(header)
        if visitor is not None:
            return _bucket(visitor)
    elif header and peer is not None:
        _warn_ignored_header(str(peer))
    if peer is not None:
        return _bucket(peer)
    return peer_text or "unknown"


# --- the FastAPI dependency ----------------------------------------------------------------

def _redis_client() -> "redis.Redis":
    from app.services.query_cache import get_redis_client  # imported late: it connects on first use

    return get_redis_client()


def rate_limit(name: str, limit: Limit, window: float) -> Callable[[Request], None]:
    """
    Build a rate limiter for ``limit`` hits per ``window`` seconds per caller.

    ``limit`` is usually a callable reading a setting; it is read on every request. Counting
    happens in Redis when RATE_LIMIT_STORE is "redis" (the default) and Redis is reachable, and
    in this process's memory otherwise.

    The result works as a FastAPI dependency, or can be called directly with the request (to
    count only requests that passed validation). The in-memory limiter is exposed as
    ``.limiter`` so tests can reset it.
    """
    from app.services.query_cache import redis_in_backoff, report_redis_failure

    memory = SlidingWindowLimiter(limit, window)
    shared = RedisWindowLimiter(name, limit, window, _redis_client)

    def check(key: str) -> Tuple[bool, float]:
        if settings.rate_limit_store == "redis" and not redis_in_backoff():
            try:
                return shared.check(key)
            except redis.RedisError as e:
                report_redis_failure(f"{name} rate limit", e)
        return memory.check(key)

    def enforce(request: Request) -> None:
        allowed, retry_after = check(client_id(request))
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many {name} requests. Please wait before trying again.",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

    enforce.limiter = memory
    enforce.shared = shared
    return enforce
