import pytest
import base64
import asyncio
from email.message import EmailMessage
from bs4 import BeautifulSoup
from unittest.mock import MagicMock

from app.utils import (
    extract_urls,
    extract_eml_body,
    extract_email_features,
    parse_raw_eml
)
from app.models import EmailRequest


def test_extract_urls():
    text = "Visit https://example.com and http://test.com for more info."
    urls = extract_urls(text)
    assert urls == ["https://example.com", "http://test.com"]


@pytest.mark.asyncio
async def test_extract_eml_body_text():
    msg = EmailMessage()
    msg.set_content("This is a plain text email.")
    result = await extract_eml_body(msg)
    assert result == "This is a plain text email.\n"


@pytest.mark.asyncio
async def test_extract_eml_body_html():
    msg = EmailMessage()
    msg.add_alternative("<html><body><p>Hello, world!</p></body></html>", subtype="html")
    result = await extract_eml_body(msg)
    # The trailing "\n" depends on how the HTML is parsed. 
    # Usually, BeautifulSoup's get_text() appends a newline at the end.
    assert result.strip() == "Hello, world!"


@pytest.mark.asyncio
async def test_extract_email_features():
    # Prepare a small HTML body with a link
    encoded_body = base64.b64encode(
        b"<html><a href='https://example.com'>Click here</a></html>"
    ).decode()

    email_mock = EmailRequest(
        subject="Test Subject",
        body=encoded_body,
        sender="sender@example.com",
        reply_to="reply@example.com",
        customerId="12345",
        attachments=[]
    )

    features = await extract_email_features(email_mock)

    assert features["subject"] == "Test Subject"

    # Confirm the body_preview
    assert features["body_preview"] == "<html><a href='https://example.com'>Click here</a></html>"[:200]

    # Check that the link is extracted
    assert "https://example.com" in features["links"]

    # Check that domains are extracted
    assert features["sender_domain"] == "example.com"
    assert features["reply_to_domain"] == "example.com"

    # Check that we have a valid email_hash
    assert "email_hash" in features
    assert len(features["email_hash"]) == 64  # typical length of a SHA-256 hex

    # Check customerId
    assert features["customerId"] == "12345"

    # Ensure _full_body_for_embedding is present but in-memory only
    assert "_full_body_for_embedding" in features


@pytest.mark.asyncio
async def test_parse_raw_eml():
    raw_email = """From: sender@example.com
Subject: Test Email
Content-Type: text/plain

This is a test email."""
    eml_bytes = raw_email.encode()

    parsed_email = await parse_raw_eml(eml_bytes)
    assert parsed_email["subject"] == "Test Email"
    assert parsed_email["sender"] == "sender@example.com"

    decoded_body = base64.b64decode(parsed_email["body"]).decode("utf-8")
    assert decoded_body.strip() == "This is a test email."


@pytest.mark.asyncio
async def test_parse_raw_eml_html():
    raw_email = """From: sender@example.com
Subject: Test Email
Content-Type: text/html

<html><body><p>This is a test email.</p></body></html>
"""
    eml_bytes = raw_email.encode()
    parsed_email = await parse_raw_eml(eml_bytes)

    assert parsed_email["subject"] == "Test Email"
    assert parsed_email["sender"] == "sender@example.com"

    decoded_body = base64.b64decode(parsed_email["body"]).decode("utf-8")
    # The trailing newline can appear depending on how the HTML is parsed
    assert decoded_body.strip() == "<html><body><p>This is a test email.</p></body></html>"
