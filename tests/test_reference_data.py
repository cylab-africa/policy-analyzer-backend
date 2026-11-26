"""Checks on the country reference data that seeds the maps and the Sources page.

These tests need no database or API keys. They catch the mistakes that are easy to make
when the JSON files are edited by hand: a missing country, a mistyped status, a broken link.
"""
import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "app" / "data"

LAW_STATUSES = {"Enacted", "Draft", "Lite", "No Law"}
MALABO_STATUSES = {"Ratified", "Signed", "No"}
DIGITAL_ID_STATUSES = {"Implemented", "Partial", "Planning", "No Data"}
AU_MEMBERS = 55


def load(name):
    with open(DATA_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def laws():
    return load("countries_law_data.json")


@pytest.fixture(scope="module")
def digital_ids():
    return load("accurate-digital-id-data.json")


def test_every_au_member_has_a_record_in_each_file(laws, digital_ids):
    assert len(laws) == AU_MEMBERS
    assert len(digital_ids) == AU_MEMBERS


def test_countries_are_unique_and_match_between_files(laws, digital_ids):
    law_codes = [r["isoAlpha2"] for r in laws]
    id_codes = [r["isoAlpha2"] for r in digital_ids]
    assert len(set(law_codes)) == len(law_codes), "duplicate isoAlpha2 in law data"
    assert len(set(id_codes)) == len(id_codes), "duplicate isoAlpha2 in digital ID data"
    assert set(law_codes) == set(id_codes)


def test_law_statuses_are_known(laws):
    for record in laws:
        assert record["dataProtectionLegislationStatus"] in LAW_STATUSES, record["country"]
        assert record["malaboConventionStatus"] in MALABO_STATUSES, record["country"]


def test_digital_id_statuses_are_known(digital_ids):
    for record in digital_ids:
        assert record["status"] in DIGITAL_ID_STATUSES, record["country"]


def test_links_are_well_formed(laws, digital_ids):
    for record in laws + digital_ids:
        for link in record.get("links") or []:
            parsed = urlparse(link["url"])
            assert parsed.scheme in {"http", "https"} and parsed.netloc, (record["country"], link["url"])
            assert link["title"].strip(), (record["country"], link["url"])
