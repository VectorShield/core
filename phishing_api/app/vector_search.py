import uuid
import logging
import os
from functools import lru_cache

from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct
from sentence_transformers import SentenceTransformer

from .database import client
from .models import EmailRequest
from .utils import extract_email_features
from .config import COLLECTION_NAME
from .logging_config import setup_logging  # Import logging setup
from .config import MODEL_PATH


# MODEL_PATH = "models/all-MiniLM-L6-v2"

# 🔹 Initialize Logger
logger = setup_logging()

# MODEL_PATH = "models/all-MiniLM-L6-v2"
logger = logging.getLogger("phishing_api")

# Load model from disk if it exists, otherwise download it
if os.path.exists(MODEL_PATH):
    logger.info(f"Loading model from {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)
else:
    logger.info("Downloading model from the internet...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.save(MODEL_PATH)  # Save for future use

@lru_cache(maxsize=1000)
def get_cached_embedding(text: str):
    """
    Returns an embedding for the given text, cached to reduce redundant computations.
    """
    return model.encode([text], show_progress_bar=False, convert_to_numpy=True).tolist()[0]

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
        f"Queued {label} email: {feats['subject']} "
        f"(ID={email_id}, custId={feats.get('customerId')})"
    )
    return f"✅ Queued {label} email: {feats['subject']}"

def check_email_similarity(email_feats: dict):
    """
    Searches Qdrant for similar emails and computes a phishing score based on similarities.
    Returns (phishing_score, reasons, closest_label).
    """
    from .utils import get_email_vector_text  # to avoid circular import

    vector_text = get_email_vector_text(email_feats)
    vector_embedding = get_cached_embedding(vector_text)

    # Optional filter for same customerId if present
    filter_ = None
    if email_feats.get("customerId"):
        filter_ = Filter(
            must=[FieldCondition(key="customerId", match=MatchValue(value=email_feats["customerId"]))]
        )

    # Query Qdrant
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector_embedding,
        query_filter=filter_,
        limit=5
    )

    phishing_score = 0.0
    phishing_matches = []
    legit_matches = []
    max_similarity = 0.0
    closest_label = None

    for r in results:
        sim = r.score
        lbl = r.payload.get("label", "unknown")

        if sim > max_similarity:
            max_similarity = sim
            closest_label = lbl

        # Basic weighting heuristic
        if lbl == "phishing":
            phishing_score += 50 * sim
            phishing_matches.append(r.payload.get("subject", "Unknown"))
        elif lbl == "legitimate":
            phishing_score -= 20 * sim
            legit_matches.append(r.payload.get("subject", "Unknown"))

    # Clamp final phishing score to 0-100
    phishing_score = max(min(phishing_score, 100), 0)
    phishing_score = int(phishing_score)

    reasons = []
    if phishing_matches:
        reasons.append(f"Similar to known phishing emails: {phishing_matches}")
    if not phishing_matches and not legit_matches:
        reasons.append("No strong phishing or legitimate indicators found.")

    return phishing_score, reasons, closest_label
