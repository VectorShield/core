import uuid
import logging
import os
import torch
import numpy as np
from functools import lru_cache
from transformers import AutoModel, AutoTokenizer, DebertaV2Tokenizer  # Import DebertaV2Tokenizer

from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct
from .database import client
from .models import EmailRequest
from .utils import extract_email_features
from .config import COLLECTION_NAME, MODEL_PATH, MODEL_NAME
from .logging_config import setup_logging  # Import logging setup

# 🔹 Initialize Logger
logger = setup_logging()
logger = logging.getLogger("phishing_api")

# 🔹 Ensure the correct tokenizer is used for DeBERTa
if "deberta" in MODEL_NAME.lower():
    TokenizerClass = DebertaV2Tokenizer
else:
    TokenizerClass = AutoTokenizer  # Default for other models like DistilBERT

# 🔹 Load Transformer Model (Supports DeBERTa, DistilBERT, or other models)
if os.path.exists(MODEL_PATH):
    logger.info(f"✅ Loading model from {MODEL_PATH}")
    tokenizer = TokenizerClass.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
else:
    logger.info(f"⚠️ Model not found at {MODEL_PATH}. Downloading {MODEL_NAME}...")
    tokenizer = TokenizerClass.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    # Save model locally to prevent repeated downloads
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    logger.info(f"✅ Model saved at: {MODEL_PATH}")

# 🔹 Ensure model is in evaluation mode
model.eval()

@lru_cache(maxsize=1000)
def get_cached_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**inputs).last_hidden_state[:, 0, :]  # Use CLS token

    return output.squeeze().tolist()  # ✅ Ensures it's a flat list of floats


def store_email(email: EmailRequest, label: str, batch_queue) -> str:
    """
    Queues the email for batch upsert in Qdrant. Uses a unique ID based on
    hash + label to avoid duplicates.
    """
    feats = extract_email_features(email)
    feats["label"] = label

    email_id_str = feats["email_hash"] + label
    email_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, email_id_str))

    # Build a single text string for the embedding
    vector_text = f"{feats['subject']} {feats['body_preview']} {' '.join(feats['links'])}"
    vector_embedding = get_cached_embedding(vector_text)

    # Create PointStruct and append to queue
    point = PointStruct(id=email_id, vector=vector_embedding, payload=feats)
    batch_queue.append(point)

    logger.info(
        f"📥 Queued {label} email: {feats['subject']} "
        f"(ID={email_id}, custId={feats.get('customerId')})"
    )
    return f"✅ Queued {label} email: {feats['subject']}"

def check_email_similarity(email_feats: dict):
    """
    Searches Qdrant for similar emails and computes a phishing score based on similarities.
    Returns (phishing_score, reasons, closest_label).
    """
    from .utils import get_email_vector_text  # Avoid circular import

    try:
        vector_text = get_email_vector_text(email_feats)
        vector_embedding = get_cached_embedding(vector_text)

        # Debugging: Ensure vector format is correct
        if not isinstance(vector_embedding, list) or not all(isinstance(x, float) for x in vector_embedding):
            logger.error(f"❌ Invalid vector format: {vector_embedding}")
            return 0, ["Error: Invalid vector format"], "Unknown"

        # Query Qdrant
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector_embedding,
            limit=20
        )

        phishing_score = 0.0
        phishing_matches = []
        legit_matches = []
        max_similarity = 0.0
        closest_label = "Unknown"

        for r in results:
            sim = r.score
            lbl = r.payload.get("label", "unknown")

            if sim > max_similarity:
                max_similarity = sim
                closest_label = lbl

            if lbl == "phishing":
                phishing_score += 50 * sim
                phishing_matches.append(r.payload.get("subject", "Unknown"))
            elif lbl == "legitimate":
                phishing_score -= 20 * sim
                legit_matches.append(r.payload.get("subject", "Unknown"))

        phishing_score = max(min(phishing_score, 100), 0)
        phishing_score = int(phishing_score)

        reasons = []
        if phishing_matches:
            reasons.append(f"Similar to known phishing emails: {phishing_matches}")
        if not phishing_matches and not legit_matches:
            reasons.append("No strong phishing or legitimate indicators found.")

        return phishing_score, reasons, closest_label

    except Exception as e:
        logger.error(f"❌ Qdrant search failed: {e}")
        return 0, ["Error searching Qdrant"], "Unknown"
