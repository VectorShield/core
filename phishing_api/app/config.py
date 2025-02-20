"""
Global configuration settings and constants.
"""
import os

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s - %(message)s"

API_TITLE = "Phishing Detection API"
API_VERSION = "1.0.0"

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.117.177:6333")
COLLECTION_NAME = "emails"

# model
MODEL_PATH = os.getenv("MODEL_PATH", "models/all-MiniLM-L6-v2")

# Batch Upsert
BATCH_SIZE = 10
FLUSH_INTERVAL = 5  # seconds
