import logging
from fastapi import APIRouter, HTTPException
from ..models import EmailRequest
from ..vector_search import store_email
from ..batch_upsert import batch_queue

logger = logging.getLogger("phishing_api")

insert_router = APIRouter()

@insert_router.post("/insert")
async def insert_email(email: EmailRequest):
    """
    Insert an email into the Qdrant vector store asynchronously.
    The `type` field must be either 'phishing' or 'legitimate'.
    """
    logger.info(f"[/insert] subject={email.subject}")
    typ = email.type.lower() if email.type else ""
    if typ not in ["phishing", "legitimate"]:
        raise HTTPException(status_code=400, detail="Invalid email type.")

    # Check if store_email is async
    if callable(store_email) and hasattr(store_email, "__call__"):
        if hasattr(store_email, "__code__") and store_email.__code__.co_flags & 0x80:
            msg = await store_email(email, typ, batch_queue)  # Await if it's async
        else:
            msg = store_email(email, typ, batch_queue)  # Call normally if it's sync
    else:
        raise TypeError("store_email is not callable.")

    return {"message": msg}
