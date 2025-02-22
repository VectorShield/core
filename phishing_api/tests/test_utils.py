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
    get_email_vector_text,
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
    assert result == "Hello, world!\n"

@pytest.mark.asyncio
async def test_extract_email_features():
    email_mock = EmailRequest(
        subject="Test Subject",
        body=base64.b64encode("<html><a href='https://example.com'>Click here</a></html>".encode()).decode(),
        sender="sender@example.com",
        reply_to="reply@example.com",
        customerId="12345",
        attachments=[]
    )

    features = await extract_email_features(email_mock)
    assert features["subject"] == "Test Subject"
    assert features["body_preview"] == "<html><a href='https://example.com'>Click here</a></html>"[:200]
    assert "https://example.com" in features["links"]
    assert features["sender_domain"] == "example.com"
    assert features["reply_to_domain"] == "example.com"
    assert "email_hash" in features
    assert features["customerId"] == "12345"

def test_get_email_vector_text():
    email_feats = {
        "subject": "Test Email",
        "body_preview": "Hello, world!",
        "links": ["https://example.com", "http://test.com"]
    }
    vector_text = get_email_vector_text(email_feats)
    assert "Test Email" in vector_text
    assert "Hello, world!" in vector_text
    assert "https://example.com" in vector_text
    assert "http://test.com" in vector_text

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
    assert base64.b64decode(parsed_email["body"]).decode("utf-8") == "This is a test email."

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
    assert decoded_body == "<html><body><p>This is a test email.</p></body></html>\n"
