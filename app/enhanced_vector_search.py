import uuid
import logging
import os
import asyncio
import torch
import numpy as np
import hashlib
import re
from functools import lru_cache
from typing import List, Dict, Optional, Tuple
from transformers import AutoModel, AutoTokenizer, DebertaV2Tokenizer
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct
from .database import client
from .models import EmailRequest
from .utils import extract_email_features
from .config import COLLECTION_NAME, MODEL_PATH, MODEL_NAME, BAD_PROB_THRESHOLD
from .logging_config import setup_logging

# Import the original vector search for fallback
from .vector_search import (
    embed_text_cached, create_aggregated_embedding, 
    check_email_similarity as original_check_similarity
)

# Initialize Logger
logger = setup_logging()
logger = logging.getLogger("enhanced_phishing_api")

class EnhancedSpamFilter:
    """Enhanced spam filter with adaptive thresholds and ensemble methods."""
    
    def __init__(self):
        self.base_threshold = BAD_PROB_THRESHOLD  # Default 0.5
        self.confidence_multipliers = {
            'Very High': 0.8,
            'High': 0.9,
            'Low': 1.2,
            'Very Low': 1.4
        }
        
        # Suspicious patterns for rule-based boost
        self.urgency_keywords = [
            'urgent', 'immediate', 'expires', 'limited time', 'act now',
            'hurry', 'deadline', 'final notice', 'last chance', 'expires today'
        ]
        
        self.suspicious_patterns = [
            'click here', 'claim now', 'verify account', 'update payment',
            'suspended account', 'unusual activity', 'confirm identity',
            '$$$', 'free money', 'guaranteed', 'risk free', 'winner',
            'congratulations', 'you have won', 'cash prize'
        ]
        
        self.sender_red_flags = [
            'noreply', 'no-reply', 'donotreply', 'automated', 'system',
            'security', 'alert', 'notification'
        ]

    def calculate_confidence_level(self, bad_score: int, good_sim: float, bad_sim: float) -> str:
        """Calculate confidence level based on score and similarity patterns."""
        total_sim = good_sim + bad_sim
        
        if total_sim < 0.1:
            return 'Very Low'
        
        # High confidence if clear separation
        if bad_score >= 80 or bad_score <= 20:
            return 'Very High'
        elif bad_score >= 70 or bad_score <= 30:
            return 'High'
        elif bad_score >= 60 or bad_score <= 40:
            return 'Low'
        else:
            return 'Very Low'

    def extract_rule_based_features(self, email_feats: dict) -> Dict[str, float]:
        """Extract rule-based features for scoring boost."""
        features = {}
        
        # Combine text for analysis
        subject = email_feats.get('subject', '').lower()
        body_preview = email_feats.get('body_preview', '').lower()
        sender = email_feats.get('sender', '').lower()
        full_text = f"{subject} {body_preview}".lower()
        
        # Urgency keywords
        urgency_count = sum(1 for keyword in self.urgency_keywords if keyword in full_text)
        features['urgency_score'] = min(urgency_count * 5, 20)  # Max 20 points
        
        # Suspicious patterns
        suspicious_count = sum(1 for pattern in self.suspicious_patterns if pattern in full_text)
        features['suspicious_score'] = min(suspicious_count * 8, 25)  # Max 25 points
        
        # Sender analysis
        sender_suspicious = sum(1 for flag in self.sender_red_flags if flag in sender)
        features['sender_score'] = min(sender_suspicious * 10, 15)  # Max 15 points
        
        # HTML to text ratio (if available)
        html_tags = len(re.findall(r'<[^>]+>', body_preview))
        if html_tags > 5:
            features['html_score'] = min(html_tags * 2, 10)  # Max 10 points
        else:
            features['html_score'] = 0
        
        # Subject line analysis
        if len(subject) > 50:
            features['subject_length_penalty'] = 5
        elif any(char in subject for char in ['!', '$', '?']):
            features['punctuation_score'] = 5
        else:
            features['subject_length_penalty'] = 0
            features['punctuation_score'] = 0
        
        return features

    def calculate_rule_boost(self, email_feats: dict) -> int:
        """Calculate rule-based boost for borderline scores."""
        rule_features = self.extract_rule_based_features(email_feats)
        
        total_boost = sum(rule_features.values())
        # Cap the boost to avoid false positives
        return min(total_boost, 30)

    def get_adaptive_threshold(self, confidence: str, good_sim: float, bad_sim: float) -> float:
        """Calculate adaptive threshold based on confidence and similarity distribution."""
        base_threshold = self.base_threshold
        
        # Adjust based on confidence
        confidence_adjustment = self.confidence_multipliers.get(confidence, 1.0)
        
        # Adjust based on data availability
        total_sim = good_sim + bad_sim
        if total_sim < 0.5:  # Low similarity data
            data_adjustment = 1.1  # More conservative
        else:
            data_adjustment = 1.0
        
        adjusted_threshold = base_threshold * confidence_adjustment * data_adjustment
        
        # Keep within reasonable bounds
        return max(0.4, min(0.7, adjusted_threshold))

    async def enhanced_check_email_similarity(self, email_feats: dict) -> Tuple[int, List[str], str, str]:
        """
        Enhanced email similarity check with adaptive thresholds and rule-based boosting.
        Returns: (score, reasons, prediction, confidence)
        """
        try:
            # Get original ML-based score
            original_score, reasons, original_prediction = await original_check_similarity(email_feats)
            
            # Extract similarity sums from reasons for confidence calculation
            good_sim = 0.0
            bad_sim = 0.0
            for reason in reasons:
                if "good_similarity_sum=" in reason:
                    good_sim = float(reason.split("=")[1])
                elif "bad_similarity_sum=" in reason:
                    bad_sim = float(reason.split("=")[1])
            
            # Calculate confidence
            confidence = self.calculate_confidence_level(original_score, good_sim, bad_sim)
            
            # Start with original score
            enhanced_score = original_score
            enhancement_reasons = []
            
            # Apply rule-based boost for borderline scores (40-65 range)
            if 40 <= original_score <= 65:
                rule_boost = self.calculate_rule_boost(email_feats)
                if rule_boost > 0:
                    enhanced_score = min(100, original_score + rule_boost)
                    enhancement_reasons.append(f"Rule-based boost: +{rule_boost} points")
                    enhancement_reasons.append(f"Enhanced score: {original_score} → {enhanced_score}")
            
            # Apply adaptive threshold
            adaptive_threshold = self.get_adaptive_threshold(confidence, good_sim, bad_sim)
            threshold_percentage = int(adaptive_threshold * 100)
            
            # Final prediction based on adaptive threshold
            enhanced_prediction = "bad" if enhanced_score >= threshold_percentage else "good"
            
            # Compile enhanced reasons
            enhanced_reasons = reasons + [
                f"Confidence level: {confidence}",
                f"Adaptive threshold: {threshold_percentage}% (was {int(self.base_threshold * 100)}%)",
                f"Final enhanced score: {enhanced_score}",
                f"Enhanced prediction: {enhanced_prediction}"
            ] + enhancement_reasons
            
            # Log enhancement details
            if enhanced_score != original_score or enhanced_prediction != original_prediction:
                logger.info(f"🔧 Enhanced scoring applied: {original_score}→{enhanced_score}, "
                           f"prediction: {original_prediction}→{enhanced_prediction}, "
                           f"confidence: {confidence}")
            
            return enhanced_score, enhanced_reasons, enhanced_prediction, confidence
            
        except Exception as e:
            logger.error(f"❌ Enhanced similarity check failed: {e}")
            # Fallback to original method
            original_score, reasons, prediction = await original_check_similarity(email_feats)
            return original_score, reasons + [f"Enhancement error: {e}"], prediction, "Low"

    async def ensemble_prediction(self, email_feats: dict, huggingface_score: Optional[int] = None) -> Tuple[int, List[str], str, str]:
        """
        Ensemble prediction combining multiple approaches.
        If huggingface_score is provided, uses ensemble weighting.
        """
        # Get enhanced score from our system
        enhanced_score, reasons, prediction, confidence = await self.enhanced_check_email_similarity(email_feats)
        
        if huggingface_score is not None and 40 <= enhanced_score <= 65:
            # Ensemble weighting for borderline cases
            ensemble_score = int(0.7 * enhanced_score + 0.3 * huggingface_score)
            ensemble_reasons = reasons + [
                f"Ensemble applied (40-65 range): 0.7 * {enhanced_score} + 0.3 * {huggingface_score} = {ensemble_score}"
            ]
            
            # Recalculate prediction with ensemble score
            threshold_percentage = int(self.get_adaptive_threshold(confidence, 0, 0) * 100)
            ensemble_prediction = "bad" if ensemble_score >= threshold_percentage else "good"
            
            return ensemble_score, ensemble_reasons, ensemble_prediction, confidence
        
        return enhanced_score, reasons, prediction, confidence

# Global instance
enhanced_filter = EnhancedSpamFilter()

# Public interface functions
async def enhanced_check_email_similarity(email_feats: dict) -> Tuple[int, List[str], str, str]:
    """Enhanced email similarity check - main public interface."""
    return await enhanced_filter.enhanced_check_email_similarity(email_feats)

async def ensemble_email_prediction(email_feats: dict, huggingface_score: Optional[int] = None) -> Tuple[int, List[str], str, str]:
    """Ensemble prediction - public interface."""
    return await enhanced_filter.ensemble_prediction(email_feats, huggingface_score)