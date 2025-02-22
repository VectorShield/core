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
# MODEL_PATH = os.getenv("MODEL_PATH", "models/deberta-v3-base")
# MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/deberta-v3-base")
# MODEL_DIMENSION = int(os.getenv("MODEL_DIMENSION", "768"))
# MODEL_NAME = "microsoft/deberta-v3-base"  # Or "distilbert-base-uncased"


MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/all-MiniLM-L6-v2")
MODEL_PATH = os.getenv("MODEL_PATH", "models/all-MiniLM-L6-v2")
MODEL_DIMENSION = int(os.getenv("MODEL_DIMENSION", "384"))


# MODEL_NAME = os.getenv("MODEL_NAME", "Intel/dynamic_tinybert")
# MODEL_PATH = os.getenv("MODEL_PATH", "models/dynamic_tinybert")
# MODEL_DIMENSION = int(os.getenv("MODEL_DIMENSION", "768"))
# Total Emails Tested: 1865
# Total Spam Emails Tested: 465
# Total Ham Emails Tested: 1400
# Correct Classifications: 1418
# False Positives: 0 (0.00%)
# False Negatives: 447 (23.97%)
# Overall Accuracy: 76.03%

# Batch Upsert
BATCH_SIZE = 10
FLUSH_INTERVAL = 5  # seconds
