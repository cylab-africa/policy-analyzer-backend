"""
Operator key for batch jobs.

Every request already needs the shared API_KEY, but the website holds that key too, so it cannot
tell the operator apart from a visitor. The /api/admin routes, changes to the country reference
data, listing every batch job and deleting any job therefore also need ADMIN_API_KEY in the
X-Admin-Key header. That key is never given to the website. While it is unset, or set to the same
value as API_KEY, those actions are switched off.
"""
import hashlib
import hmac
from typing import Optional

from app.core.config import settings


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def admin_enabled() -> bool:
    """True when a usable admin key is configured (set, and different from the shared API key)."""
    configured = settings.admin_api_key
    if not configured:
        return False
    # Reusing the shared key would hand these rights to the website and everyone it serves
    return not hmac.compare_digest(_digest(configured), _digest(settings.api_key or ""))


def is_admin(supplied: Optional[str]) -> bool:
    """
    True only if admin access is enabled and ``supplied`` equals ADMIN_API_KEY.

    Both sides are hashed before the constant-time comparison, so any string, including
    non-ASCII text, compares safely.
    """
    if not supplied or not admin_enabled():
        return False
    return hmac.compare_digest(_digest(supplied), _digest(settings.admin_api_key))
