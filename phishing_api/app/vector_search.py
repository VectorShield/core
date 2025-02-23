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
# from .utils import get_email_vector_text

# 🔹 Initialize Logger
logger = setup_logging()
logger = logging.getLogger("phishing_api")

# 🔹 Ensure the correct tokenizer is used for DeBERTa
if "deberta" in MODEL_NAME.lower():
    TokenizerClass = DebertaV2Tokenizer
else:
    TokenizerClass = AutoTokenizer  # Default for other models like DistilBERT

# Load or download model
if os.path.exists(MODEL_PATH):
    logger.info(f"✅ Loading model from {MODEL_PATH}")
    tokenizer = TokenizerClass.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
else:
    logger.info(f"⚠️ Model not found at {MODEL_PATH}. Downloading {MODEL_NAME}...")
    tokenizer = TokenizerClass.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    logger.info(f"✅ Model saved at: {MODEL_PATH}")

# 🔹 Ensure model is in evaluation mode
model.eval()

def chunk_text(text: str, chunk_size=256, overlap=50) -> list[str]:
    """
    Simple approach: chunk the text by characters, with optional overlap.
    For a more robust approach, chunk by tokens (using tokenizer).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)  # e.g. 256 - 50 = 206 step
    return chunks

def embed_text(text: str) -> torch.Tensor:
    """
    Embed a single string. Returns a torch vector of shape [hidden_dim].
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**inputs).last_hidden_state[:, 0, :]  # CLS token
    return output.squeeze(0)  # shape [hidden_dim]

def create_aggregated_embedding(full_text: str) -> list[float]:
    """
    Splits the text into multiple chunks, embeddings each chunk,
    then returns the average embedding as a plain Python list[float].
    """
    # 1) Chunk text
    chunks = chunk_text(full_text, chunk_size=256, overlap=50)
    embeddings = []

    for c in chunks:
        # Optionally skip empty or whitespace chunks
        if c.strip():
            emb = embed_text(c)
            embeddings.append(emb)

    if not embeddings:
        # fallback: embed just an empty string or so
        return embed_text("").tolist()

    # 2) Average all chunk embeddings
    stacked = torch.stack(embeddings, dim=0)  # shape [num_chunks, hidden_dim]
    avg_embedding = torch.mean(stacked, dim=0)  # shape [hidden_dim]

    return avg_embedding.tolist()

async def get_cached_embedding(text: str):
    """Async wrapper for embedding generation."""
    return await asyncio.to_thread(sync_get_cached_embedding, text)

def sync_get_cached_embedding(text: str):
    """Synchronous part of generating an embedding."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**inputs).last_hidden_state[:, 0, :]  # Use [CLS] token
    return output.squeeze().tolist()


async def store_email(email: EmailRequest, label: str, batch_queue) -> str:
    """
    Asynchronously queues the email for batch upsert in Qdrant. Uses a chunked embedding of the entire body
    without storing the full body in Qdrant.
    """
    feats = await extract_email_features(email)

    if not isinstance(feats, dict):
        raise TypeError(f"extract_email_features returned {type(feats)}, expected dict")

    feats["label"] = label

    # Use the email_hash + label as unique ID
    email_id_str = feats["email_hash"] + label
    email_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, email_id_str))

    # --- 1) Build an aggregated embedding from the entire body in memory
    full_body_str = feats.pop("_full_body_for_embedding", None)  # remove it from final feats
    if full_body_str:
        # chunk-embed the entire body
        vector_embedding = create_aggregated_embedding(full_body_str)
    else:
        # fallback: embed just subject + preview if no full body
        fallback_text = f"{feats['subject']} {feats['body_preview']}"
        vector_embedding = await get_cached_embedding(fallback_text)

    # --- 2) Create Qdrant point struct
    point = PointStruct(
        id=email_id,
        vector=vector_embedding,
        payload=feats  # 'feats' no longer has the full body
    )
    batch_queue.append(point)

    logger.info(f"📥 Queued {label} email: {feats['subject']} (ID={email_id})")
    return f"✅ Queued {label} email: {feats['subject']}"


async def check_email_similarity(email_feats: dict):
    """
    Updated version of check_email_similarity that uses chunk-based embedding
    instead of get_email_vector_text.
    """
    try:
        # Attempt to retrieve the full body from feats if available
        full_body = email_feats.pop("_full_body_for_embedding", None)

        # If we have the full body in memory, create a chunk-based embedding:
        if full_body and full_body.strip():
            vector_embedding = create_aggregated_embedding(full_body)
        else:
            # fallback: embed the subject + short preview
            fallback_text = f"{email_feats.get('subject','')} {email_feats.get('body_preview','')}"
            # you'll need an async call if you want to re-use the original "get_cached_embedding"
            # or just do it inline with embed_text like so:
            vector_embedding = embed_text(fallback_text).tolist()

        # Now we can query Qdrant
        results = await client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector_embedding,
            limit=50
        )

        sum_phish_sim = 0.0
        sum_legit_sim = 0.0

        for r in results:
            sim = r.score
            label = r.payload.get("label", "unknown")
            if label == "phishing":
                sum_phish_sim += sim
            elif label == "legitimate":
                sum_legit_sim += sim

        # Avoid divide-by-zero
        epsilon = 1e-7
        denom = sum_phish_sim + sum_legit_sim + epsilon
        phish_prob = sum_phish_sim / denom

        # 0..100 score
        phishing_score = int(round(phish_prob * 100))
        closest_label = "phishing" if phish_prob >= 0.5 else "legitimate"

        reasons = []
        if sum_phish_sim > sum_legit_sim:
            reasons.append(f"Sum of phishing similarity {sum_phish_sim:.2f} > legit {sum_legit_sim:.2f}")
        else:
            reasons.append(f"Sum of legitimate similarity {sum_legit_sim:.2f} >= phishing {sum_phish_sim:.2f}")

        if sum_phish_sim == 0.0 and sum_legit_sim == 0.0:
            reasons.append("No nearest neighbors labeled phishing or legitimate found.")
        elif abs(sum_phish_sim - sum_legit_sim) < 0.01:
            reasons.append("Phishing vs. legit similarity is nearly identical.")

        return phishing_score, reasons, closest_label

    except Exception as e:
        logger.error(f"❌ Qdrant search failed: {e}")
        return 0, ["Error searching Qdrant"], "Unknown"








