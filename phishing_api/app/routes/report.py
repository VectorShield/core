import logging
from fastapi import APIRouter, HTTPException
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from ..models import EmailRequest
from ..database import client
from ..vector_search import get_cached_embedding
from ..utils import extract_email_features, get_email_vector_text
from ..config import COLLECTION_NAME

logger = logging.getLogger("phishing_api")

report_router = APIRouter()

@report_router.post("/report_false_positive")
def report_false_positive(email: EmailRequest):
    """
    Removes an email from the store if it was incorrectly marked as phishing.
    """
    logger.info(f"[/report_false_positive] subject={email.subject}")
    feats = extract_email_features(email)

    vector_text = get_email_vector_text(feats)
    vec = get_cached_embedding(vector_text)

    filt = None
    if feats.get("customerId"):
        filt = Filter(must=[FieldCondition(key="customerId", match=MatchValue(value=feats["customerId"]))])

    res = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        query_filter=filt,
        limit=1
    )
    if res:
        e_id = res[0].id
        client.delete(collection_name=COLLECTION_NAME, points_selector=[e_id])
        logger.info(f"Removed false positive: {feats['subject']} (ID={e_id})")
        return {"message": f"Removed false positive email: {feats['subject']}"}
    else:
        logger.warning("Email not found.")
        raise HTTPException(status_code=404, detail="Email not found in DB.")
