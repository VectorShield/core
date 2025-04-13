# ./tests/test_parse_eml.py

import pytest
import base64
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_parse_eml_valid_text():
    """
    Upload a valid plain-text EML file to /parse_eml and verify the response.
    """
    # Create a minimal EML with plain text
    eml_content = (
        "From: test@example.com\n"
        "Subject: Plain Text Test\n"
        "Content-Type: text/plain\n\n"
        "This is a test email."
    ).encode("utf-8")

    files = {
        "file": ("test.eml", eml_content, "message/rfc822")
    }

    response = client.post("/parse_eml", files=files)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"

    data = response.json()
    assert "email" in data, "Response JSON must contain 'email' key"
    assert data["message"] == "Parsed EML"

    email_obj = data["email"]
    assert email_obj["subject"] == "Plain Text Test"
    assert email_obj["sender"] == "test@example.com"

    # Decode the body
    decoded_body = base64.b64decode(email_obj["body"]).decode("utf-8")
    assert decoded_body.strip() == "This is a test email."

def test_parse_eml_valid_html():
    """
    Upload a valid HTML EML file to /parse_eml and verify the response.
    """
    eml_content = (
        "From: test@example.com\n"
        "Subject: HTML Test\n"
        "Content-Type: text/html\n\n"
        "<html><body><h1>Hello World</h1></body></html>"
    ).encode("utf-8")

    files = {
        "file": ("test.eml", eml_content, "message/rfc822")
    }

    response = client.post("/parse_eml", files=files)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"

    data = response.json()
    assert "email" in data
    assert data["message"] == "Parsed EML"

    email_obj = data["email"]
    assert email_obj["subject"] == "HTML Test"
    assert email_obj["sender"] == "test@example.com"

    decoded_body = base64.b64decode(email_obj["body"]).decode("utf-8")
    assert decoded_body.strip() == "<html><body><h1>Hello World</h1></body></html>"

def test_parse_eml_missing_file():
    """
    Attempt to call /parse_eml without actually uploading any file.
    Expect a 422 error (Unprocessable Entity) from FastAPI.
    """
    response = client.post("/parse_eml", files={})
    # Because the endpoint requires a file, this usually triggers a 422.
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"

def test_parse_eml_invalid_file():
    """
    Provide an invalid (non-EML) file content to /parse_eml and expect a 400 from our code.
    """
    # The route tries to parse the EML content. If it fails, it raises HTTP 400.
    files = {
        "file": ("fake.txt", b"Not a valid EML content", "text/plain")
    }

    response = client.post("/parse_eml", files=files)
    # Our route logic returns 400 if the parse fails
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    data = response.json()
    assert "Failed to parse EML" in data["detail"], "Must mention parse failure in the detail"
