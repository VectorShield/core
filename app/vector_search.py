import uuid
import logging
import os
import asyncio
import torch
import numpy as np
import hashlib
from functools import lru_cache
from typing import List, Dict, Optional
from transformers import AutoModel, AutoTokenizer, DebertaV2Tokenizer
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct
from .database import client
from .models import EmailRequest
from .utils import extract_email_features
from .config import COLLECTION_NAME, MODEL_PATH, MODEL_NAME, BAD_PROB_THRESHOLD
from .logging_config import setup_logging

# Initialize Logger
logger = setup_logging()
logger = logging.getLogger("phishing_api")

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# In-memory embedding cache (LRU cache with max 10000 entries)
embedding_cache: Dict[str, List[float]] = {}
CACHE_MAX_SIZE = 10000

# Cache performance metrics
cache_stats = {
    "hits": 0,
    "misses": 0,
    "total_requests": 0
}

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

def _get_cache_key(text: str) -> str:
    """Generate a cache key for text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def _manage_cache_size():
    """Remove oldest entries if cache exceeds max size."""
    if len(embedding_cache) > CACHE_MAX_SIZE:
        # Remove 20% of entries to avoid frequent cache management
        remove_count = len(embedding_cache) // 5
        keys_to_remove = list(embedding_cache.keys())[:remove_count]
        for key in keys_to_remove:
            del embedding_cache[key]
        
        # Log cache performance
        hit_rate = cache_stats["hits"] / max(cache_stats["total_requests"], 1) * 100
        logger.info(f"Cache cleaned: removed {remove_count} entries. "
                   f"Cache hit rate: {hit_rate:.1f}% ({cache_stats['hits']}/{cache_stats['total_requests']})")

def get_cache_performance() -> Dict[str, float]:
    """Get cache performance statistics."""
    total_requests = cache_stats["total_requests"]
    if total_requests == 0:
        return {"hit_rate": 0.0, "size": len(embedding_cache), "total_requests": 0}
    
    hit_rate = (cache_stats["hits"] / total_requests) * 100
    return {
        "hit_rate": hit_rate,
        "size": len(embedding_cache),
        "total_requests": total_requests,
        "hits": cache_stats["hits"],
        "misses": cache_stats["misses"]
    }

def chunk_text_intelligent(text: str, max_tokens=384) -> List[str]:
    """
    Intelligently chunk text based on sentences and token limits.
    More efficient than fixed character chunking.
    """
    if not text or len(text.strip()) == 0:
        return [""]
    
    # For short texts, return as single chunk
    if len(text) <= max_tokens:
        return [text]
    
    # Split by sentences first
    import re
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed limit, start new chunk
        if len(current_chunk) + len(sentence) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text[:max_tokens]]

def embed_text_cached(text: str) -> List[float]:
    """Embed text with caching support."""
    cache_stats["total_requests"] += 1
    
    if not text or not text.strip():
        return [0.0] * model.config.hidden_size
    
    cache_key = _get_cache_key(text)
    
    # Check cache first
    if cache_key in embedding_cache:
        cache_stats["hits"] += 1
        return embedding_cache[cache_key]
    
    cache_stats["misses"] += 1
    
    # Generate embedding
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model(**inputs).last_hidden_state[:, 0, :].cpu()
    
    embedding = output.squeeze(0).tolist()
    
    # Cache the result
    embedding_cache[cache_key] = embedding
    _manage_cache_size()
    
    return embedding

def embed_texts_batch(texts: List[str], batch_size: int = 8) -> List[List[float]]:
    """
    Batch embed multiple texts for better GPU utilization.
    Returns list of embeddings in same order as input texts.
    """
    if not texts:
        return []
    
    embeddings = []
    cached_results = {}
    texts_to_embed = []
    indices_to_embed = []
    
    # Check cache for each text
    for i, text in enumerate(texts):
        if not text or not text.strip():
            cached_results[i] = [0.0] * model.config.hidden_size
            continue
            
        cache_key = _get_cache_key(text)
        if cache_key in embedding_cache:
            cached_results[i] = embedding_cache[cache_key]
        else:
            texts_to_embed.append(text)
            indices_to_embed.append(i)
    
    # Batch embed uncached texts
    if texts_to_embed:
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            batch_indices = indices_to_embed[i:i + batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs).last_hidden_state[:, 0, :].cpu()
            
            # Store results
            for j, idx in enumerate(batch_indices):
                embedding = outputs[j].tolist()
                cached_results[idx] = embedding
                
                # Cache the result
                cache_key = _get_cache_key(texts_to_embed[i + j])
                embedding_cache[cache_key] = embedding
    
    # Reconstruct embeddings in original order
    embeddings = [cached_results[i] for i in range(len(texts))]
    _manage_cache_size()
    
    return embeddings

def create_aggregated_embedding(full_text: str) -> List[float]:
    """Generate an optimized average embedding over intelligent text chunks."""
    if not full_text or not full_text.strip():
        return [0.0] * model.config.hidden_size
    
    # Use intelligent chunking
    chunks = chunk_text_intelligent(full_text, max_tokens=400)
    
    # Use batch embedding for efficiency
    embeddings = embed_texts_batch(chunks)
    
    if not embeddings:
        return [0.0] * model.config.hidden_size
    
    # Calculate weighted average (longer chunks get more weight)
    total_weight = 0
    weighted_embedding = np.zeros(len(embeddings[0]))
    
    for i, chunk in enumerate(chunks):
        weight = len(chunk.strip())  # Weight by text length
        if weight > 0:
            weighted_embedding += np.array(embeddings[i]) * weight
            total_weight += weight
    
    if total_weight > 0:
        weighted_embedding /= total_weight
    
    return weighted_embedding.tolist()

async def get_cached_embedding(text: str) -> List[float]:
    """Async wrapper for cached embedding generation."""
    return await asyncio.to_thread(embed_text_cached, text)

async def store_email(
    email: EmailRequest, 
    label: str,               # "good" or "bad"
    sub_label: str,           # e.g. "spam", "transactional", ...
    batch_queue
) -> str:
    """
    Queues email for batch upsert in Qdrant.
    'label' is the main_label ('good' or 'bad'),
    'sub_label' is the finer classification ('spam', 'transactional', etc.).
    """
    feats = await extract_email_features(email)
    if not isinstance(feats, dict):
        raise TypeError(f"extract_email_features returned {type(feats)}, expected dict")

    # Store both labels:
    feats["label"] = label           # main label, e.g. "good" or "bad"
    feats["sub_label"] = sub_label   # finer label, e.g. "spam", "transactional"

    # We build a stable UUID using the email_hash plus the label combination
    import uuid
    email_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, feats["email_hash"] + label + sub_label))

    full_body_str = feats.pop("_full_body_for_embedding", None)

    if full_body_str:
        vector_embedding = create_aggregated_embedding(full_body_str)
    else:
        # fallback: just embed the subject + preview
        fallback_text = f"{feats['subject']} {feats['body_preview']}"
        vector_embedding = await get_cached_embedding(fallback_text)

    # Add the data to the queue
    from qdrant_client.http.models import PointStruct
    batch_queue.append(
        PointStruct(
            id=email_id,
            vector=vector_embedding,
            payload=feats
        )
    )

    logger.info(f"📥 Queued {label.upper()} email [{sub_label}]: {feats['subject']} (ID={email_id})")
    return f"✅ Queued {label.upper()} email [{sub_label}]: {feats['subject']}"

async def check_email_similarity(email_feats: dict):
    """
    Retrieve nearest neighbors from Qdrant and compute a 'bad_score' percentage
    (like the old phishing_score). The final label is 'bad' if >= 50%, otherwise 'good'.
    """
    try:
        full_body = email_feats.pop("_full_body_for_embedding", None)
        if full_body and full_body.strip():
            vector_embedding = create_aggregated_embedding(full_body)
        else:
            # fallback to cached embedding
            fallback_text = f"{email_feats.get('subject','')} {email_feats.get('body_preview','')}"
            vector_embedding = embed_text_cached(fallback_text)

        # Use higher limit for better coverage
        adaptive_limit = min(200, max(50, len(embedding_cache) // 50))
        
        results = await client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector_embedding,
            limit=adaptive_limit,
            score_threshold=0.005  # Lower threshold to catch more subtle similarities
        )

        # Simple similarity-based scoring
        good_similarity_sum = 0.0
        bad_similarity_sum = 0.0

        # Track best matches for each category
        best_good_match = {"score": 0.0, "sub_label": "unknown"}
        best_bad_match = {"score": 0.0, "sub_label": "unknown"}
        
        # Simple accumulation of similarity scores
        for r in results:
            sim = r.score
            lbl = r.payload.get("label", "unknown")
            sbl = r.payload.get("sub_label", "unknown")
            
            if lbl == "good":
                good_similarity_sum += sim
                if sim > best_good_match["score"]:
                    best_good_match = {"score": sim, "sub_label": sbl}
                    
            elif lbl == "bad":
                bad_similarity_sum += sim
                if sim > best_bad_match["score"]:
                    best_bad_match = {"score": sim, "sub_label": sbl}

        # Simple ratio-based scoring
        total_similarity = good_similarity_sum + bad_similarity_sum
        if total_similarity > 0:
            bad_prob = bad_similarity_sum / total_similarity
        else:
            bad_prob = 0.5  # Neutral when no matches
        
        bad_score = int(round(bad_prob * 100))
        closest_label = "bad" if bad_prob >= BAD_PROB_THRESHOLD else "good"
        
        # Determine overall best match
        if best_bad_match["score"] > best_good_match["score"]:
            top_match_label = "bad"
            top_match_sub_label = best_bad_match["sub_label"]
            top_match_score = best_bad_match["score"]
        else:
            top_match_label = "good"
            top_match_sub_label = best_good_match["sub_label"]
            top_match_score = best_good_match["score"]

        reasons = [
            f"good_similarity_sum={good_similarity_sum:.3f}",
            f"bad_similarity_sum={bad_similarity_sum:.3f}",
            f"bad_probability={bad_prob:.3f}",
            f"threshold={BAD_PROB_THRESHOLD:.3f}",
            f"matches_found={len(results)}"
        ]
        
        if top_match_label:
            reasons.append(f"Top match => {top_match_label}, sub_label={top_match_sub_label}, score={top_match_score:.3f}")
        
        # Add category-specific insights
        if best_good_match["score"] > 0:
            reasons.append(f"Best good match: {best_good_match['sub_label']} (score={best_good_match['score']:.3f})")
        if best_bad_match["score"] > 0:
            reasons.append(f"Best bad match: {best_bad_match['sub_label']} (score={best_bad_match['score']:.3f})")

        # If we have no matches at all
        if len(results) == 0:
            reasons.append("No similar emails found in database")
            return 50, reasons, "unknown"  # Neutral score when no data

        return bad_score, reasons, closest_label

    except Exception as e:
        logger.error(f"❌ Qdrant search failed: {e}")
        return 0, [f"Error searching Qdrant: {e}"], "Unknown"

