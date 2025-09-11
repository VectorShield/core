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

        # 3) Derive enhanced confidence level from phishing score and reasoning
        confidence_factors = []
        reasons_text = " ".join(reasons)
        
        # Extract confidence indicators from reasoning
        matches_found = 0
        adaptive_threshold = 0.6
        
        for reason in reasons:
            if "matches_found=" in reason:
                matches_found = int(reason.split("=")[1])
            elif "adaptive_threshold=" in reason:
                adaptive_threshold = float(reason.split("=")[1])
        
        # Score-based confidence
        if phishing_score >= 85 or phishing_score <= 15:
            confidence_factors.append("extreme_score")
        elif phishing_score >= 75 or phishing_score <= 25:
            confidence_factors.append("high_score")
        elif phishing_score >= 65 or phishing_score <= 35:
            confidence_factors.append("medium_score")
        else:
            confidence_factors.append("low_score")
        
        # Match quantity confidence
        if matches_found >= 30:
            confidence_factors.append("many_matches")
        elif matches_found >= 15:
            confidence_factors.append("adequate_matches")
        elif matches_found >= 5:
            confidence_factors.append("few_matches")
        else:
            confidence_factors.append("very_few_matches")
        
        # Determine final confidence
        if "extreme_score" in confidence_factors and "many_matches" in confidence_factors:
            conf = "Very High"
        elif ("extreme_score" in confidence_factors or 
              ("high_score" in confidence_factors and "many_matches" in confidence_factors)):
            conf = "High"
        elif ("high_score" in confidence_factors or 
              ("medium_score" in confidence_factors and "adequate_matches" in confidence_factors)):
            conf = "Medium"
        elif "very_few_matches" in confidence_factors:
            conf = "Very Low"
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