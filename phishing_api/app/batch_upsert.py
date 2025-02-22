"""
Handles batch upsert logic for Qdrant.
"""
import asyncio
import time
import logging
from collections import deque

from .database import client
from .config import BATCH_SIZE, FLUSH_INTERVAL, COLLECTION_NAME

logger = logging.getLogger("phishing_api")

batch_queue = deque()

async def batch_upsert():
    """
    Background task that periodically upserts batched points into Qdrant.
    """
    last_upsert_time = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - last_upsert_time
        if len(batch_queue) >= BATCH_SIZE or (elapsed >= FLUSH_INTERVAL and len(batch_queue) > 0):
            points = []
            while batch_queue:
                points.append(batch_queue.popleft())

            await client.upsert(COLLECTION_NAME, points)
            logger.info(f"✅ Upserted {len(points)} points to Qdrant.")
            last_upsert_time = time.perf_counter()

        await asyncio.sleep(1)

async def init_batch_upsert():
    """
    Called on application startup to begin the background upsert process.
    """
    asyncio.create_task(batch_upsert())
