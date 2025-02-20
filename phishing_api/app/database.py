"""
Initialize the Qdrant client and ensure the required collection exists.
"""
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import VectorParams, Distance

from .config import QDRANT_URL, COLLECTION_NAME

logger = logging.getLogger("phishing_api")

# Initialize Qdrant client
client = QdrantClient(QDRANT_URL)

# Ensure collection is created
try:
    client.get_collection(collection_name=COLLECTION_NAME)
    logger.info(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
except UnexpectedResponse as e:
    if "Collection" in str(e) and "doesn" in str(e):
        logger.info(f"Collection '{COLLECTION_NAME}' does not exist. Creating it now...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    else:
        raise
