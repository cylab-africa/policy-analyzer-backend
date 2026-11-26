"""
Per-job delete tokens.

The API has one shared key and no user accounts, so "who owns this job" cannot be answered
from the request. Instead each batch job gets a random token when it is created. The token is
returned to the creator once and only a hash of it is stored, so a leaked database does not
reveal working tokens. Deleting a job requires the token.
"""
import hashlib
import hmac
import secrets
from typing import Optional


def new_token() -> str:
    """A fresh unguessable token (256 bits)."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_matches(token: Optional[str], stored_hash: Optional[str]) -> bool:
    """True only if ``token`` hashes to ``stored_hash``. Compared in constant time."""
    if not token or not stored_hash:
        return False
    return hmac.compare_digest(hash_token(token), stored_hash)
