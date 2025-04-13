import logging
from fastapi import APIRouter, HTTPException
from ..models import EmailRequest
from ..vector_search import store_email
from ..batch_upsert import batch_queue

logger = logging.getLogger("phishing_api")

insert_router = APIRouter()

# Define all valid sub-labels and map them to their main label (good or bad)
VALID_TYPES = {
    # good
    "transactional": "good",
    "business": "good",
    "newsletter": "good",
    "subscription": "good",
    "notification": "good",
    "marketing": "good",
    # bad
    "spam": "bad",
    "bounced": "bad",
}

@insert_router.post("/insert")
async def insert_email(email: EmailRequest):
    """
    Insert an email into the Qdrant vector store asynchronously.
    The `type` field must be one of:
      - transactional, business, newsletter, subscription, notification, marketing (all 'good')
      - spam, bounced (all 'bad')
    """
    logger.info(f"[/insert] subject={email.subject}")

    if not email.type:
        raise HTTPException(status_code=400, detail="No email type provided.")

    sub_label = email.type.lower().strip()

    if sub_label not in VALID_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid email type '{email.type}'. "
                f"Must be one of: {', '.join(VALID_TYPES.keys())}."
            ),
        )

    main_label = VALID_TYPES[sub_label]  # "good" or "bad"

    # The store_email function expects the second argument as the “label” to store
    # in Qdrant, but we also want to pass the sub_label. We'll do that by combining
    # them into a small dict or tuple that store_email can handle.
    # For backward-compat, the store_email function uses "label" in the Qdrant payload,
    # so we will treat that as our main_label, and store sub_label in the payload as well.

    msg = await store_email(
        email=email,
        label=main_label,         # old top-level label
        sub_label=sub_label,      # new finer category
        batch_queue=batch_queue
    )

    return {"message": msg}
