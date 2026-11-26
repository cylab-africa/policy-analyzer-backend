"""The /api/admin routes can change prompts and tabs, so only the operator may use them. The
website holds the shared API key, so that key alone must never be enough."""
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.core.config import settings
from app.routes import admin

ADMIN_KEY = "operator-only-admin-key"
CLEAR = "/api/admin/prompts/cache/clear"


class FakePromptManager:
    def __init__(self):
        self.cleared = 0

    def clear_cache(self):
        self.cleared += 1


@pytest.fixture()
def prompts(monkeypatch):
    manager = FakePromptManager()
    monkeypatch.setattr(admin, "get_prompt_manager", lambda *args, **kwargs: manager)
    monkeypatch.setattr(settings, "admin_api_key", ADMIN_KEY)
    return manager


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(admin.router)
    return TestClient(app)


@pytest.mark.parametrize("headers", [{}, {"X-Admin-Key": "wrong"}, {"X-Admin-Key": "test-api-key"}])
def test_refused_without_the_admin_key(client, prompts, headers):
    assert client.post(CLEAR, headers=headers).status_code == 401
    assert prompts.cleared == 0


def test_the_operator_is_let_in(client, prompts):
    assert client.post(CLEAR, headers={"X-Admin-Key": ADMIN_KEY}).status_code == 200
    assert prompts.cleared == 1


@pytest.mark.parametrize("configured", [None, "", "test-api-key"])
def test_routes_are_off_without_a_separate_admin_key(client, prompts, monkeypatch, configured):
    monkeypatch.setattr(settings, "admin_api_key", configured)
    for supplied in ("test-api-key", ADMIN_KEY):
        assert client.post(CLEAR, headers={"X-Admin-Key": supplied}).status_code == 401
    assert prompts.cleared == 0


def test_every_admin_route_has_the_check():
    """A route added later without the dependency would be open to the shared key."""
    routes = [route for route in admin.router.routes if isinstance(route, APIRoute)]
    assert routes
    unguarded = [
        route.path for route in routes
        if admin.verify_admin_key not in [d.call for d in route.dependant.dependencies]
    ]
    assert unguarded == []
