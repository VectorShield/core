import uuid
import logging
import os
import asyncio
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, DebertaV2Tokenizer
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct
from .database import client
from .models import EmailRequest
from .utils import extract_email_features
from .config import COLLECTION_NAME, MODEL_PATH, MODEL_NAME
from .logging_config import setup_logging

# Initialize Logger
logger = setup_logging()
logger = logging.getLogger("phishing_api")

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Ensure the correct tokenizer is used for DeBERTa
TokenizerClass = DebertaV2Tokenizer if "deberta" in MODEL_NAME.lower() else AutoTokenizer

# Load or download model
if os.path.exists(MODEL_PATH):
    logger.info(f"✅ Loading model from {MODEL_PATH}")
    tokenizer = TokenizerClass.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH).to(device)
else:
    logger.info(f"⚠️ Model not found at {MODEL_PATH}. Downloading {MODEL_NAME}...")
    tokenizer = TokenizerClass.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    logger.info(f"✅ Model saved at: {MODEL_PATH}")

# Ensure model is in evaluation mode
model.eval()

def chunk_text(text: str, chunk_size=256, overlap=50) -> list[str]:
    """Splits text into chunks with optional overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def embed_text(text: str) -> torch.Tensor:
    """Embed a single string using the model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output = model(**inputs).last_hidden_state[:, 0, :].to("cpu")  # Move back to CPU
    return output.squeeze(0)

def create_aggregated_embedding(full_text: str) -> list[float]:
    """Generate an average embedding over text chunks."""
    chunks = chunk_text(full_text, chunk_size=256, overlap=50)
    embeddings = [embed_text(c) for c in chunks if c.strip()]
    
    if not embeddings:
        return embed_text("").tolist()
    
    avg_embedding = torch.mean(torch.stack(embeddings), dim=0)  # Average over all chunks
    return avg_embedding.tolist()

async def get_cached_embedding(text: str):
    """Async wrapper for embedding generation."""
    return await asyncio.to_thread(sync_get_cached_embedding, text)

def sync_get_cached_embedding(text: str):
    """Synchronous embedding generation with comments."""

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate embeddings using the model while ensuring gradients are not computed
    with torch.no_grad():
        output = model(**inputs).last_hidden_state[:, 0, :].to("cpu")

    # Return the embedding as a list of floats
    return output.squeeze().tolist()

async def store_email(email: EmailRequest, label: str, batch_queue) -> str:
    """Queues email for batch upsert in Qdrant."""
    feats = await extract_email_features(email)
    if not isinstance(feats, dict):
        raise TypeError(f"extract_email_features returned {type(feats)}, expected dict")

    feats["label"] = label
    email_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, feats["email_hash"] + label))
    full_body_str = feats.pop("_full_body_for_embedding", None)

    if full_body_str:
        vector_embedding = create_aggregated_embedding(full_body_str)
    else:
        fallback_text = f"{feats['subject']} {feats['body_preview']}"
        vector_embedding = await get_cached_embedding(fallback_text)

    batch_queue.append(PointStruct(id=email_id, vector=vector_embedding, payload=feats))
    logger.info(f"📥 Queued {label} email: {feats['subject']} (ID={email_id})")
    return f"✅ Queued {label} email: {feats['subject']}"

async def check_email_similarity(email_feats: dict):
    """Retrieve nearest neighbors and compute phishing score."""
    try:
        full_body = email_feats.pop("_full_body_for_embedding", None)
        vector_embedding = create_aggregated_embedding(full_body) if full_body and full_body.strip() else embed_text(f"{email_feats.get('subject','')} {email_feats.get('body_preview','')}").tolist()

        results = await client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector_embedding,
            limit=50,
            score_threshold=0.01
        )

        sum_phish_sim, sum_legit_sim = 0.0, 0.0
        for r in results:
            sim = r.score
            lbl = r.payload.get("label", "unknown")
            if lbl == "phishing":
                sum_phish_sim += sim
            elif lbl == "legitimate":
                sum_legit_sim += sim

        phish_prob = sum_phish_sim / (sum_phish_sim + sum_legit_sim + 1e-7)
        phishing_score = int(round(phish_prob * 100))
        closest_label = "phishing" if phish_prob >= 0.5 else "legitimate"
        reasons = [f"PhishSim={sum_phish_sim:.3f}, LegitSim={sum_legit_sim:.3f}"]

        if sum_phish_sim == 0.0 and sum_legit_sim == 0.0:
            reasons.append("No nearest neighbors labeled phishing or legitimate found.")
        elif abs(sum_phish_sim - sum_legit_sim) < 0.01:
            reasons.append("Phishing vs. legitimate similarity is nearly identical.")

        return phishing_score, reasons, closest_label
    except Exception as e:
        logger.error(f"❌ Qdrant search failed: {e}")
        return 0, ["Error searching Qdrant"], "Unknown"
