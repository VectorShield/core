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
    Includes error handling to avoid crashing if Qdrant is temporarily unreachable.
    """
    last_upsert_time = time.perf_counter()

    # Optional: set a base delay if you want a backoff on repeated failures
    base_retry_delay = 3  # seconds
    current_retry_delay = base_retry_delay

    while True:
        elapsed = time.perf_counter() - last_upsert_time
        # If we've either accumulated enough points or enough time has passed,
        # we attempt to upsert.
        if len(batch_queue) >= BATCH_SIZE or (elapsed >= FLUSH_INTERVAL and len(batch_queue) > 0):
            points = []
            while batch_queue:
                points.append(batch_queue.popleft())

            try:
                await client.upsert(COLLECTION_NAME, points)
                logger.info(f"✅ Upserted {len(points)} points to Qdrant.")
                last_upsert_time = time.perf_counter()
                # On success, reset retry delay
                current_retry_delay = base_retry_delay

            except Exception as e:
                # If we fail, we log and re-queue the points
                logger.error(f"❌ Upsert to Qdrant failed: {e}")
                logger.info(f"Re-queueing {len(points)} points and retrying...")

                # Put points back into the left side of the queue
                # so we can try again on the next iteration
                for p in reversed(points):
                    batch_queue.appendleft(p)

                # Optional: Wait longer before the next attempt (simple backoff)
                logger.info(f"Waiting {current_retry_delay} seconds before retry...")
                await asyncio.sleep(current_retry_delay)

                # Exponential backoff (cap to some max if you like)
                current_retry_delay = min(current_retry_delay * 2, 60)

        # Normal sleep loop if no batch is ready or after an attempt
        await asyncio.sleep(1)


async def init_batch_upsert():
    """
    Called on application startup to begin the background upsert process.
    """
    asyncio.create_task(batch_upsert())
