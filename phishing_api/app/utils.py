import re
import base64
import hashlib
import logging
import asyncio
import email
from email.parser import BytesParser
from email import policy
from typing import List
from bs4 import BeautifulSoup

from .models import EmailRequest

logger = logging.getLogger("phishing_api")

def extract_urls(text: str) -> List[str]:
    """Extract all http/https URLs from a string."""
    return re.findall(r"https?://[^\s\"<>]+", text)

async def extract_eml_body(msg: email.message.Message) -> str:
    """
    Extract the text body from an email.message.Message object asynchronously.
    """
    def sync_extract():
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                cdisp = str(part.get("Content-Disposition"))
                if ctype == "text/plain" and "attachment" not in cdisp:
                    return part.get_payload(decode=True).decode(errors="replace")
                elif ctype == "text/html" and "attachment" not in cdisp:
                    return BeautifulSoup(
                        part.get_payload(decode=True).decode(errors="replace"),
                        "html.parser"
                    ).get_text()
        return msg.get_payload(decode=True).decode(errors="replace")
    return await asyncio.to_thread(sync_extract)

async def extract_email_features(email: EmailRequest):
    """
    Extract key features, including a short body_preview, domain mismatch, links, etc.
    WITHOUT storing the entire body in the final payload.
    """
    try:
        decoded_body_bytes = base64.b64decode(email.body)
    except Exception:
        logger.exception("Failed to decode base64 email body.")
        decoded_body_bytes = b""

    # We'll return this so the embedding function can see the entire body,
    # but we won't store it in the final payload. 
    body_str = decoded_body_bytes.decode(errors="replace")

    # Only store a short snippet in Qdrant
    body_preview = body_str[:2000]

    subject = email.subject or ""
    sender = email.sender or "unknown@domain"
    reply_to = email.reply_to or sender
    customer_id = email.customerId

    sender_domain = sender.split("@")[-1] if "@" in sender else "unknown"
    reply_to_domain = reply_to.split("@")[-1] if "@" in reply_to else "unknown"
    sender_reply_mismatch = (sender_domain.lower() != reply_to_domain.lower())

    # Extract links
    soup = BeautifulSoup(body_str, "html.parser")
    html_links = [a["href"] for a in soup.find_all("a", href=True)]
    text_links = extract_urls(body_str)
    merged_links = list(set(html_links + text_links))

    def compute_hash(links_joined: str):
        email_string = subject + body_preview + links_joined + sender_domain
        return hashlib.sha256(email_string.encode()).hexdigest()

    email_hash = compute_hash("".join(merged_links))

    # Collect link domains
    link_domains = []
    for link in merged_links:
        match = re.match(r"^https?://([^/]+)/?", link)
        if match:
            link_domains.append(match.group(1).lower())
    link_domains = list(set(link_domains))

    # Return a dictionary with minimal stored data
    # But also pass back the entire 'body_str' for embedding usage
    feats = {
        "subject": subject,
        "body_preview": body_preview,
        "links": merged_links,
        "link_domains": link_domains,
        "sender_domain": sender_domain,
        "reply_to_domain": reply_to_domain,
        "sender_reply_mismatch": sender_reply_mismatch,
        "attachments": email.attachments or [],
        "email_hash": email_hash,
        "customerId": customer_id,
        # Add 'full_body' just in memory, not for final storage:
        "_full_body_for_embedding": body_str  # We'll remove this before storing in Qdrant
    }

    return feats

# ./app/utils.py

async def parse_raw_eml(eml_bytes: bytes):
    """
    Parse a raw EML file asynchronously.
    Returns a dict with subject, base64-encoded body, and sender.
    """
    msg = BytesParser(policy=policy.default).parsebytes(eml_bytes)
    subject = msg["subject"] or "No Subject"
    sender = msg["from"] or "Unknown Sender"
    body_str = await extract_eml_body(msg)

    # Optional: Strict Check
    # If we detect an obviously invalid EML or "junk" content, raise an error.
    # For instance, say we want to ensure the sender is not "Unknown Sender" 
    # AND the body has at least some non-whitespace text:
    if sender == "Unknown Sender":
        raise ValueError("Invalid EML content – missing sender and body text.")

    return {
        "subject": subject,
        "body": base64.b64encode(body_str.encode("utf-8")).decode("utf-8"),
        "sender": sender
    }

