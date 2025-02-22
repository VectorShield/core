# utils.py

import re
import base64
import hashlib
import logging
import asyncio
from bs4 import BeautifulSoup
import email
from email.parser import BytesParser
from email import policy
from typing import List
from .models import EmailRequest

logger = logging.getLogger("phishing_api")

def extract_urls(text: str) -> List[str]:
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
    Extract key features (links, domains, body preview) from the EmailRequest.
    """
    try:
        decoded_body_bytes = base64.b64decode(email.body)
    except Exception:
        logger.exception("Failed to decode base64 email body.")
        decoded_body_bytes = b""

    body_str = decoded_body_bytes.decode(errors="replace")
    body_preview = body_str[:200]

    subject = email.subject
    sender = email.sender
    reply_to = email.reply_to or sender
    customer_id = email.customerId

    sender_domain = sender.split("@")[-1] if "@" in sender else "unknown"
    reply_to_domain = reply_to.split("@")[-1] if "@" in reply_to else "unknown"

    async def extract_links():
        soup = BeautifulSoup(body_str, "html.parser")
        html_links = [a['href'] for a in soup.find_all('a', href=True)]
        text_links = extract_urls(body_str)
        return list(set(html_links + text_links))

    async def compute_hash(links_joined: str):
        email_string = subject + body_preview + links_joined + sender_domain
        return hashlib.sha256(email_string.encode()).hexdigest()

    merged_links = await extract_links()
    email_hash = await compute_hash("".join(merged_links))

    return {
        "subject": subject,
        "body_preview": body_preview,
        "links": merged_links,
        "sender_domain": sender_domain,
        "reply_to_domain": reply_to_domain,
        "attachments": email.attachments or [],
        "email_hash": email_hash,
        "customerId": customer_id,
    }

def get_email_vector_text(email_feats: dict) -> str:
    return f"{email_feats['subject']} {email_feats['body_preview']} {' '.join(email_feats['links'])}"

async def parse_raw_eml(eml_bytes: bytes):
    """
    Parse a raw EML file asynchronously.
    Returns a dict with subject, base64-encoded body, and sender.
    """
    # Parse the email synchronously
    msg = BytesParser(policy=policy.default).parsebytes(eml_bytes)

    # Pull out metadata
    subject = msg["subject"] or "No Subject"
    sender = msg["from"] or "Unknown Sender"

    # Actually get the body text asynchronously
    body_str = await extract_eml_body(msg)

    return {
        "subject": subject,
        "body": base64.b64encode(body_str.encode("utf-8")).decode("utf-8"),
        "sender": sender
    }
