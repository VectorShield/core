import logging
from fastapi import APIRouter
from ..models import EmailRequest, AnalyzeResponse
from ..utils import extract_email_features
from ..vector_search import check_email_similarity

logger = logging.getLogger("phishing_api")

analyze_router = APIRouter()

@analyze_router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_email(email: EmailRequest):
    """
    Analyze an email by extracting features (including the full body for in-memory embedding)
    and then checking similarity against Qdrant via a chunk-based embedding approach.
    """
    logger.info(f"[/analyze] subject={email.subject}")

    try:
        # 1) Extract features from the email (stores full body in memory but not in Qdrant)
        feats = await extract_email_features(email)

        # 2) Compute phishing score, reasons, and closest label
        phishing_score, reasons, closest_label = await check_email_similarity(feats)

        # 3) Derive confidence level from phishing score
        if phishing_score >= 80:
            conf = "High"
        elif phishing_score >= 60:
            conf = "Medium"
        else:
            conf = "Low"

        logger.info(
            f"✅ Analyzed -> score={phishing_score}, conf={conf}, label={closest_label}, {reasons=}"
        )

        # 4) Return an AnalyzeResponse
        return AnalyzeResponse(
            phishing_score=phishing_score,
            confidence_level=conf,
            closest_match=closest_label,
            reasons=reasons
        )

    except Exception as e:
        logger.exception(f"❌ Exception occurred while analyzing email: {e}")

        return AnalyzeResponse(
            phishing_score=0,
            confidence_level="Unknown",
            closest_match="Unknown",
            reasons=[f"Error: {str(e)}"]
        )