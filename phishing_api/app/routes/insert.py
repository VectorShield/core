import logging
from fastapi import APIRouter, HTTPException
from ..models import EmailRequest
from ..vector_search import store_email
from ..batch_upsert import batch_queue

logger = logging.getLogger("phishing_api")

insert_router = APIRouter()

@insert_router.post("/insert")
def insert_email(email: EmailRequest):
    """
    Insert an email into the Qdrant vector store.
    The `type` field must be either 'phishing' or 'legitimate'.
    """
    logger.info(f"[/insert] subject={email.subject}")
    typ = email.type.lower() if email.type else ""
    if typ not in ["phishing", "legitimate"]:
        raise HTTPException(status_code=400, detail="Invalid email type.")

    msg = store_email(email, typ, batch_queue)
    return {"message": msg}
