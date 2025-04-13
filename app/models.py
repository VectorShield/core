from pydantic import BaseModel
from typing import List, Optional

class EmailRequest(BaseModel):
    subject: str
    body: str  # base64-encoded
    sender: str
    reply_to: Optional[str] = None
    attachments: Optional[List[str]] = []
    type: Optional[str] = None
    customerId: Optional[str] = None

class AnalyzeResponse(BaseModel):
    phishing_score: int
    confidence_level: str
    closest_match: Optional[str]
    reasons: List[str]
