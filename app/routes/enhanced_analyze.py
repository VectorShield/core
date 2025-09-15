import logging
from fastapi import APIRouter
from typing import Optional
from ..models import EmailRequest
from ..utils import extract_email_features
from ..vector_search import check_email_similarity
from ..enhanced_vector_search import enhanced_check_email_similarity, ensemble_email_prediction
from pydantic import BaseModel

logger = logging.getLogger("enhanced_phishing_api")

enhanced_analyze_router = APIRouter()

class EnhancedAnalyzeResponse(BaseModel):
    """Response model for enhanced analysis with comparison."""
    original_score: int
    original_prediction: str
    original_confidence: str
    enhanced_score: int
    enhanced_prediction: str
    enhanced_confidence: str
    ensemble_score: Optional[int] = None
    ensemble_prediction: Optional[str] = None
    improvement_applied: bool
    rule_boost: int
    adaptive_threshold: int
    reasons: list[str]

@enhanced_analyze_router.post("/analyze/enhanced", response_model=EnhancedAnalyzeResponse)
async def analyze_email_enhanced(email: EmailRequest):
    """
    Enhanced email analysis with side-by-side comparison of original and improved models.
    Shows exactly what improvements were applied.
    """
    logger.info(f"[/analyze/enhanced] subject={email.subject}")

    try:
        # 1) Extract features from the email
        feats = await extract_email_features(email)

        # 2) Get original model results
        original_score, original_reasons, original_prediction = await check_email_similarity(feats)
        
        # Calculate original confidence (simplified)
        if original_score >= 80 or original_score <= 20:
            original_confidence = "Very High"
        elif original_score >= 70 or original_score <= 30:
            original_confidence = "High"
        elif original_score >= 60 or original_score <= 40:
            original_confidence = "Low"
        else:
            original_confidence = "Very Low"

        # 3) Get enhanced model results
        enhanced_score, enhanced_reasons, enhanced_prediction, enhanced_confidence = await enhanced_check_email_similarity(feats)

        # 4) Calculate improvement metrics
        improvement_applied = enhanced_score != original_score or enhanced_prediction != original_prediction
        
        # Extract rule boost and adaptive threshold from reasons
        rule_boost = 0
        adaptive_threshold = 50
        
        for reason in enhanced_reasons:
            if "Rule-based boost:" in reason and "+" in reason:
                try:
                    rule_boost = int(reason.split("+")[1].split()[0])
                except:
                    pass
            elif "Adaptive threshold:" in reason:
                try:
                    adaptive_threshold = int(reason.split(":")[1].split("%")[0].strip())
                except:
                    pass

        # 5) Optional ensemble prediction (if you had HuggingFace score)
        ensemble_score = None
        ensemble_prediction = None

        logger.info(
            f"✅ Enhanced Analysis -> "
            f"original={original_score}({original_prediction}) -> "
            f"enhanced={enhanced_score}({enhanced_prediction}) "
            f"improvement={improvement_applied}"
        )

        # 6) Return comparison results
        return EnhancedAnalyzeResponse(
            original_score=original_score,
            original_prediction=original_prediction,
            original_confidence=original_confidence,
            enhanced_score=enhanced_score,
            enhanced_prediction=enhanced_prediction,
            enhanced_confidence=enhanced_confidence,
            ensemble_score=ensemble_score,
            ensemble_prediction=ensemble_prediction,
            improvement_applied=improvement_applied,
            rule_boost=rule_boost,
            adaptive_threshold=adaptive_threshold,
            reasons=enhanced_reasons
        )

    except Exception as e:
        logger.exception(f"❌ Exception in enhanced analysis: {e}")

        return EnhancedAnalyzeResponse(
            original_score=0,
            original_prediction="error",
            original_confidence="Unknown",
            enhanced_score=0,
            enhanced_prediction="error", 
            enhanced_confidence="Unknown",
            improvement_applied=False,
            rule_boost=0,
            adaptive_threshold=50,
            reasons=[f"Error: {str(e)}"]
        )

@enhanced_analyze_router.post("/analyze/ensemble")
async def analyze_email_ensemble(email: EmailRequest, huggingface_score: Optional[int] = None):
    """
    Ensemble analysis combining multiple model predictions.
    Provide huggingface_score parameter for best results.
    """
    logger.info(f"[/analyze/ensemble] subject={email.subject}, hf_score={huggingface_score}")

    try:
        # Extract features
        feats = await extract_email_features(email)

        # Get ensemble prediction
        ensemble_score, reasons, prediction, confidence = await ensemble_email_prediction(feats, huggingface_score)

        logger.info(f"✅ Ensemble Analysis -> score={ensemble_score}, prediction={prediction}, conf={confidence}")

        return {
            "ensemble_score": ensemble_score,
            "ensemble_prediction": prediction,
            "confidence": confidence,
            "huggingface_input": huggingface_score,
            "reasons": reasons
        }

    except Exception as e:
        logger.exception(f"❌ Exception in ensemble analysis: {e}")
        return {
            "ensemble_score": 0,
            "ensemble_prediction": "error",
            "confidence": "Unknown",
            "huggingface_input": huggingface_score,
            "reasons": [f"Error: {str(e)}"]
        }