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
#QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.117.177:6333")
QDRANT_URL = os.getenv("QDRANT_URL", "http://172.17.0.1:6333")
COLLECTION_NAME = "emails"
BAD_PROB_THRESHOLD = float(os.getenv("BAD_PROB_THRESHOLD", "0.52"))

# Security settings
DISABLE_CSP = os.getenv("DISABLE_CSP", "true").lower() == "true"

# model
# MODEL_PATH = os.getenv("MODEL_PATH", "models/deberta-v3-base")
# MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/deberta-v3-base")
# MODEL_DIMENSION = int(os.getenv("MODEL_DIMENSION", "768"))
# MODEL_NAME = "microsoft/deberta-v3-base"  # Or "distilbert-base-uncased"
# 📊 Test Summary:
# Total Emails Tested: 570
# Correct Classifications: 468 (82.11%)
# False Positives: 0 (0.00%)
# False Negatives: 102 (17.89%)

# 🔍 Confidence Level Breakdown:
# High Confidence: 30 (5.26%)
# Medium Confidence: 13 (2.28%)
# Low Confidence: 527 (92.46%)

# 🔎 Similarity Statistics:
# Avg PhishSim: 8.538
# Avg LegitSim: 41.449

MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-small-en-v1.5")
MODEL_PATH = os.getenv("MODEL_PATH", "models/bge-small-en-v1.5")
MODEL_DIMENSION = int(os.getenv("MODEL_DIMENSION", "384"))
# Total Emails Tested: 1865
# Total Spam Emails Tested: 465
# Total Ham Emails Tested: 1400
# Correct Classifications: 1468
# False Positives: 0 (0.00%)
# False Negatives: 397 (21.29%)
# Overall Accuracy: 78.71%
# 🔍 Confidence Level Statistics:
# High Confidence: 68 (3.65%)
# Medium Confidence: 1 (0.05%)
# Low Confidence: 1796 (96.30%


# MODEL_NAME = os.getenv("MODEL_NAME", "Intel/dynamic_tinybert")
# MODEL_PATH = os.getenv("MODEL_PATH", "models/dynamic_tinybert")
# MODEL_DIMENSION = int(os.getenv("MODEL_DIMENSION", "768"))
# 📊 Test Summary:
# Total Emails Tested: 570
# Correct Classifications: 444 (77.89%)
# False Positives: 0 (0.00%)
# False Negatives: 126 (22.11%)

# 🔍 Confidence Level Breakdown:
# High Confidence: 3 (0.53%)
# Medium Confidence: 24 (4.21%)
# Low Confidence: 543 (95.26%)

# 🔎 Similarity Statistics:
# Avg PhishSim: 9.884
# Avg LegitSim: 39.848

# Batch Upsert
BATCH_SIZE = 40
FLUSH_INTERVAL = 10  # seconds
