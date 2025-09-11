import logging
from fastapi import APIRouter, HTTPException
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from ..models import EmailRequest
from ..database import client
from ..utils import extract_email_features
from ..config import COLLECTION_NAME
from ..vector_search import create_aggregated_embedding, embed_text_cached

logger = logging.getLogger("phishing_api")

report_router = APIRouter()

@report_router.post("/report_false_positive")
async def report_false_positive(email: EmailRequest):
    """
    Removes the first matching vector from Qdrant that corresponds to the given email.
    Uses a chunk-based (or fallback) embedding approach to locate the nearest neighbor.
    """
    logger.info(f"[/report_false_positive] subject={email.subject}")

    # 1) Extract features (including the full body in-memory if you have that)
    feats = await extract_email_features(email)

    # 2) Either chunk-embed the full body or fallback to subject+preview
    full_body = feats.pop("_full_body_for_embedding", None)
    if full_body and full_body.strip():
        embedding = create_aggregated_embedding(full_body)  # returns a list[float]
    else:
        # minimal fallback text
        fallback_text = f"{feats.get('subject','')} {feats.get('body_preview','')}"
        embedding = embed_text_cached(fallback_text)  # single-pass embedding with caching

    # 3) Build an optional filter based on customerId
    filt = None
    if feats.get("customerId"):
        filt = Filter(
            must=[FieldCondition(
                key="customerId",
                match=MatchValue(value=feats["customerId"])
            )]
        )

    # 4) Search Qdrant for the closest match (limit=1)
    res = await client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        query_filter=filt,
        limit=1
    )

    # 5) If found, delete it
    if res:
        e_id = res[0].id
        await client.delete(collection_name=COLLECTION_NAME, points_selector=[e_id])
        logger.info(f"✅ Removed false positive: {feats['subject']} (ID={e_id})")
        return {"message": f"Removed false positive email: {feats['subject']}"}
    else:
        logger.warning("⚠️ Email not found.")
        raise HTTPException(status_code=404, detail="Email not found in DB.")
