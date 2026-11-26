"""Batch emails are sent from the configured account, never from an address built into the code."""
import inspect

from app.core import config
from app.core.config import Settings, settings
from app.services.email_service import EmailService


def test_no_email_address_is_built_into_the_settings():
    fresh = Settings(openai_api_key="k", mongodb_connection_string="m", api_key="a")
    assert fresh.SENDER_EMAIL is None
    assert "@gmail.com" not in inspect.getsource(config)


def test_sender_defaults_to_the_account_that_logs_in(monkeypatch):
    monkeypatch.setattr(settings, "SMTP_USERNAME", "reports@example.org")
    monkeypatch.setattr(settings, "SENDER_EMAIL", None)
    assert EmailService().sender_email == "reports@example.org"


def test_a_configured_sender_is_used(monkeypatch):
    monkeypatch.setattr(settings, "SMTP_USERNAME", "reports@example.org")
    monkeypatch.setattr(settings, "SENDER_EMAIL", "noreply@example.org")
    assert EmailService().sender_email == "noreply@example.org"
