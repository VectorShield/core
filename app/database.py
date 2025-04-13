# database.py
import logging
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import VectorParams, Distance

from .config import QDRANT_URL, COLLECTION_NAME, MODEL_DIMENSION

logger = logging.getLogger("phishing_api")

# Initialize Qdrant client
client = AsyncQdrantClient(QDRANT_URL)

async def ensure_collection_exists():
    """
    Ensures the Qdrant collection is created before use.
    Call this once at startup BEFORE upserting anything.
    """
    try:
        await client.get_collection(collection_name=COLLECTION_NAME)
        logger.info(f"✅ Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
    except UnexpectedResponse as e:
        if "Collection" in str(e) and "doesn" in str(e):
            logger.info(f"⚠️ Collection '{COLLECTION_NAME}' does not exist. Creating it now...")
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=MODEL_DIMENSION, distance=Distance.COSINE),
            )
            logger.info(f"✅ Collection '{COLLECTION_NAME}' created.")
        else:
            raise
