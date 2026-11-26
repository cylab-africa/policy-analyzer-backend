"""The country reference data can be read by anyone with the API key, but only the operator can
change it. The website holds the API key and adds it to every request it passes on, so the API
key alone must not allow writes. The database is an in-memory fake."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services import info_api

ADMIN_KEY = "operator-only-admin-key"

KENYA = {
    "country": "Kenya",
    "yearLegislationEnacted": "2019",
    "malaboConventionStatus": "No",
    "dataProtectionLegislationStatus": "Enacted",
    "isoAlpha2": "KE",
}


class Result:
    def __init__(self, matched=0, deleted=0, inserted_id=None):
        self.matched_count = matched
        self.deleted_count = deleted
        self.inserted_id = inserted_id


class FakeCollection:
    def __init__(self, docs):
        self.docs = [dict(d) for d in docs]

    def _matches(self, doc, query):
        return all(doc.get(k) == v for k, v in query.items())

    def find_one(self, query, projection=None):
        for doc in self.docs:
            if self._matches(doc, query):
                return {k: v for k, v in doc.items() if k != "_id"}
        return None

    def update_one(self, query, update):
        for doc in self.docs:
            if self._matches(doc, query):
                doc.update(update["$set"])
                return Result(matched=1)
        return Result()

    def delete_one(self, query):
        before = len(self.docs)
        self.docs = [d for d in self.docs if not self._matches(d, query)]
        return Result(deleted=before - len(self.docs))

    def insert_one(self, doc):
        new_id = len(self.docs) + 1
        self.docs.append({**doc, "_id": new_id})
        return Result(inserted_id=new_id)


class FakeMongo:
    def __init__(self):
        self.collections = {"policy_law_docs": FakeCollection([KENYA]), "digital_ids_docs": FakeCollection([])}

    def get_collection(self, name):
        return self.collections[name]


@pytest.fixture()
def client(monkeypatch):
    mongo = FakeMongo()
    monkeypatch.setattr(info_api, "mongo_client", mongo)
    monkeypatch.setattr(settings, "admin_api_key", ADMIN_KEY)
    app = FastAPI()
    app.include_router(info_api.router, prefix="/api/info")
    test_client = TestClient(app)
    test_client.laws = mongo.collections["policy_law_docs"]
    return test_client


WRITES = [
    ("delete", "/api/info/policy_laws/country/KE", None),
    ("delete", "/api/info/digital_id/country/KE", None),
    ("patch", "/api/info/policy_laws/country/KE", {**KENYA, "dataProtectionLegislationStatus": "No Law"}),
    ("patch", "/api/info/digital_id/country/KE", {"country": "Kenya", "status": "Planning", "isoAlpha2": "KE"}),
    ("post", "/api/info/policy_laws", {**KENYA, "isoAlpha2": "UG", "country": "Uganda"}),
    ("post", "/api/info/digital_id", {"country": "Uganda", "status": "Planning", "isoAlpha2": "UG"}),
]


@pytest.mark.parametrize("method,path,body", WRITES)
@pytest.mark.parametrize("headers", [{}, {"X-Admin-Key": "wrong"}, {"X-Admin-Key": "test-api-key"}])
def test_writes_are_refused_without_the_admin_key(client, method, path, body, headers):
    before = [dict(d) for d in client.laws.docs]
    response = getattr(client, method)(path, json=body, headers=headers) if body else \
        getattr(client, method)(path, headers=headers)
    assert response.status_code == 403
    assert client.laws.docs == before


@pytest.mark.parametrize("method,path,body", WRITES)
def test_writes_are_refused_while_admin_access_is_off(client, monkeypatch, method, path, body):
    monkeypatch.setattr(settings, "admin_api_key", None)
    headers = {"X-Admin-Key": ADMIN_KEY}
    response = getattr(client, method)(path, json=body, headers=headers) if body else \
        getattr(client, method)(path, headers=headers)
    assert response.status_code == 403


def test_the_operator_can_change_the_data(client):
    admin = {"X-Admin-Key": ADMIN_KEY}
    updated = {**KENYA, "dataProtectionLegislationStatus": "Draft"}
    assert client.patch("/api/info/policy_laws/country/KE", json=updated, headers=admin).status_code == 200
    assert client.laws.find_one({"isoAlpha2": "KE"})["dataProtectionLegislationStatus"] == "Draft"
    assert client.delete("/api/info/policy_laws/country/KE", headers=admin).status_code == 200
    assert client.laws.find_one({"isoAlpha2": "KE"}) is None


def test_reading_needs_no_admin_key(client):
    response = client.get("/api/info/policy_laws/country/KE")
    assert response.status_code == 200
    assert response.json()["country"] == "Kenya"
