import logging
from fastapi import APIRouter
from ..models import EmailRequest, AnalyzeResponse
from ..utils import extract_email_features
from ..vector_search import check_email_similarity

logger = logging.getLogger("phishing_api")

analyze_router = APIRouter()

@analyze_router.post("/analyze", response_model=AnalyzeResponse)
def analyze_email(email: EmailRequest):
    """
    Analyze an email against the Qdrant vector store to determine a phishing score.
    """
    logger.info(f"[/analyze] subject={email.subject}")
    feats = extract_email_features(email)
    score, reasons, closest_label = check_email_similarity(feats)

    if score >= 70:
        conf = "High"
    elif score >= 40:
        conf = "Medium"
    else:
        conf = "Low"

    logger.info(f"Analyzed -> score={score}, conf={conf}, label={closest_label}")
    return AnalyzeResponse(
        phishing_score=score,
        confidence_level=conf,
        closest_match=closest_label,
        reasons=reasons
    )
