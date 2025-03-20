import os
import time
import base64
import requests
import pandas as pd
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------
# 📁 Dataset Paths (Hugging Face)
# -------------------------------
# https://huggingface.co/datasets/legacy107/spamming-email-classification
DATASET_BASE_URL = "hf://datasets/legacy107/spamming-email-classification/"
splits = {
    "train": "data/train-00000-of-00001-14eed08eb524d6f5.parquet",
    "test": "data/test-00000-of-00001-622decef5f682f6f.parquet"
}

# -------------------------------
# ⚙️ API Endpoints
# -------------------------------
INSERT_API_URL = f"{os.getenv("API_URL", "http://localhost")}/insert"
# INSERT_API_URL = "http://localhost:5000/insert"
ANALYZE_API_URL = f"{os.getenv("API_URL", "http://localhost")}/analyze"
# ANALYZE_API_URL = "http://localhost:5000/analyze"

# -------------------------------
# 📦 Load Datasets
# -------------------------------
print("Loading datasets...")
train_data = pd.read_parquet(DATASET_BASE_URL + splits["train"])
test_data = pd.read_parquet(DATASET_BASE_URL + splits["test"])
print(f"✅ Datasets loaded successfully! (Train: {len(train_data)}, Test: {len(test_data)})")

# Count Spam and Ham in Test Set
spam_count = test_data[test_data["Spam"] == 1].shape[0]
ham_count = test_data[test_data["Spam"] == 0].shape[0]

print(f"📊 Test Dataset Breakdown:")
print(f"  - Spam Emails: {spam_count}")
print(f"  - Non-Spam (Ham) Emails: {ham_count}\n")

# -------------------------------
# 📌 Helper Functions
# -------------------------------
def get_email_type(label):
    """Convert label to readable email type."""
    return "phishing" if label == 1 else "legitimate"

def parse_email_text(text):
    """Treat the entire text as the body and leave subject empty."""
    return "", text.strip()

def process_email(row):
    """Prepare and send email data to API."""
    try:
        subject, body = parse_email_text(row["Text"])
        email_type = get_email_type(row["Spam"])

        payload = {
            "subject": subject,  # Always empty
            "body": base64.b64encode(body.encode("utf-8")).decode("utf-8"),
            "sender": "unknown@example.com",
            "type": email_type
        }

        resp = requests.post(INSERT_API_URL, json=payload)
        return resp.status_code == 200
    except Exception as e:
        print(f"❌ Exception processing email: {e}")
        return False

# -------------------------------
# 🚀 Import Emails into API
# -------------------------------
print(f"📤 Importing {len(train_data)} training emails into the API...")
with ThreadPoolExecutor(max_workers=5) as executor, tqdm(total=len(train_data), desc="Importing", unit="emails") as pbar:
    futures = [executor.submit(process_email, row) for _, row in train_data.iterrows()]
    for future in as_completed(futures):
        if future.result():
            pbar.update(1)

print("\n✅ Training data import completed!")

# -------------------------------
# 🛠️ Test the API
# -------------------------------
print(f"\n🔍 Testing {len(test_data)} emails...")
time.sleep(3)

# 🔍 Initialize Metrics
total_tested = 0
correct_classifications = 0
false_positives = 0
false_negatives = 0
high_confidence = 0
medium_confidence = 0
low_confidence = 0
phish_sims = []
legit_sims = []

for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Testing", unit="emails"):
    try:
        subject, body = parse_email_text(row["Text"])
        expected_type = get_email_type(row["Spam"])

        payload = {
            "subject": subject,  # Always empty
            "body": base64.b64encode(body.encode("utf-8")).decode("utf-8"),
            "sender": "unknown@example.com"
        }

        response = requests.post(ANALYZE_API_URL, json=payload)
        response_data = response.json()

        if response.status_code == 200:
            total_tested += 1
            confidence_level = response_data.get("confidence_level", "Unknown")
            phishing_score = response_data.get("phishing_score", 0)
            predicted_type = "phishing" if phishing_score >= 70 else "legitimate"

            if confidence_level == "High":
                high_confidence += 1
            elif confidence_level == "Medium":
                medium_confidence += 1
            elif confidence_level == "Low":
                low_confidence += 1

            if predicted_type == expected_type:
                correct_classifications += 1
            elif predicted_type == "phishing" and expected_type == "legitimate":
                false_positives += 1
            elif predicted_type == "legitimate" and expected_type == "phishing":
                false_negatives += 1

            # Extract similarity scores
            reasons = response_data.get("reasons", [])
            for reason in reasons:
                match = re.search(r"PhishSim=([\d.]+).+LegitSim=([\d.]+)", reason)
                if match:
                    phish_sims.append(float(match.group(1)))
                    legit_sims.append(float(match.group(2)))

    except Exception as e:
        print(f"❌ Error analyzing email: {e}")

# -------------------------------
# 📊 Generate Final Report
# -------------------------------
if total_tested > 0:
    accuracy = (correct_classifications / total_tested) * 100
    false_positive_rate = (false_positives / total_tested) * 100
    false_negative_rate = (false_negatives / total_tested) * 100
    high_confidence_rate = (high_confidence / total_tested) * 100
    medium_confidence_rate = (medium_confidence / total_tested) * 100
    low_confidence_rate = (low_confidence / total_tested) * 100

    print("\n📊 Test Summary:")
    print(f"Total Emails Tested: {total_tested}")
    print(f"Correct Classifications: {correct_classifications} ({accuracy:.2f}%)")
    print(f"False Positives: {false_positives} ({false_positive_rate:.2f}%)")
    print(f"False Negatives: {false_negatives} ({false_negative_rate:.2f}%)")

    print("\n🔍 Confidence Level Breakdown:")
    print(f"High Confidence: {high_confidence} ({high_confidence_rate:.2f}%)")
    print(f"Medium Confidence: {medium_confidence} ({medium_confidence_rate:.2f}%)")
    print(f"Low Confidence: {low_confidence} ({low_confidence_rate:.2f}%)")

    if phish_sims and legit_sims:
        avg_phish = sum(phish_sims) / len(phish_sims)
        avg_legit = sum(legit_sims) / len(legit_sims)
        print("\n🔎 Similarity Statistics:")
        print(f"Avg PhishSim: {avg_phish:.3f}")
        print(f"Avg LegitSim: {avg_legit:.3f}")
else:
    print("\n❌ No emails were tested. Check dataset or API.")

print("\n✅ Script Execution Completed!")
