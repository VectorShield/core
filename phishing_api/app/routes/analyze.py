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
    Analyze an email against the Qdrant vector store asynchronously to determine a phishing score.
    """
    logger.info(f"[/analyze] subject={email.subject}")

    try:
        # 🔹 Call extract_email_features **without** await if it is not async
        feats = await extract_email_features(email)

        score, reasons, closest_label = await check_email_similarity(feats)  # Await similarity check

        # Ensure valid values before assigning confidence level
        if score is None:
            logger.error("❌ Error: 'score' is None")
            score = 0  # Default safe value
        if reasons is None:
            logger.error("❌ Error: 'reasons' is None")
            reasons = ["Error: No reasons provided."]
        if closest_label is None:
            logger.error("❌ Error: 'closest_label' is None")
            closest_label = "Unknown"

        # ✅ Ensure confidence level is assigned correctly
        if score >= 70:
            conf = "High"
        elif score >= 40:
            conf = "Medium"
        else:
            conf = "Low"

        logger.info(f"✅ Analyzed -> score={score}, conf={conf}, label={closest_label}")

        return AnalyzeResponse(
            phishing_score=score,
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
