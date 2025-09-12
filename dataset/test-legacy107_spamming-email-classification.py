import os
import time
import base64
import requests
import pandas as pd
import re
import csv
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
INSERT_API_URL = f"{os.getenv('API_URL', 'http://localhost:5000')}/insert"
ANALYZE_API_URL = f"{os.getenv('API_URL', 'http://localhost:5000')}/analyze"

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
    return "spam" if label == 1 else "business"

def get_expected_classification(label):
    """Convert dataset label to expected classification type for comparison"""
    return "spam" if label == 1 else "legitimate"


def parse_email_text(text):
    """Treat the entire text as the body; use an empty subject."""
    return "", text.strip()

def process_email_for_insert(row):
    """
    Prepares and sends a single email's data to the /insert endpoint.
    Returns True if inserted successfully, False otherwise.
    """
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
        return (resp.status_code == 200)
    except Exception as e:
        print(f"❌ Exception inserting email: {e}")
        return False

def analyze_email(row, idx):
    """
    Sends the email to /analyze and gathers:
    - row index (as 'EmailID')
    - expected_type
    - predicted_type
    - confidence_level
    - phishing_score
    - PhishSim / LegitSim from reasons
    - correctness (Correct / FalsePositive / FalseNegative)

    Returns a dict for CSV export & final stats.
    """
    subject, body = parse_email_text(row["Text"])
    expected_type = get_expected_classification(row["Spam"])  # "spam" or "legitimate"

    payload = {
        "subject": subject,
        "body": base64.b64encode(body.encode("utf-8")).decode("utf-8"),
        "sender": "unknown@example.com"
    }

    try:
        response = requests.post(ANALYZE_API_URL, json=payload)
        if response.status_code != 200:
            return {
                "EmailID": idx,
                "ExpectedType": expected_type,
                "PredictedType": "ERROR",
                "PhishingScore": None,
                "Confidence": "ERROR",
                "BadSim": None,
                "GoodSim": None,
                "Correctness": "AnalyzeFailed",
            }

        response_data = response.json()
        phishing_score = response_data.get("phishing_score", 0)
        confidence_level = response_data.get("confidence_level", "Unknown")
        reasons = response_data.get("reasons", [])

        # Extract "weighted_bad_score=xx, weighted_good_score=yy" from reasons
        bad_sim = None
        good_sim = None
        for reason in reasons:
            if "weighted_bad_score=" in reason:
                bad_sim = float(reason.split("=")[1])
            elif "weighted_good_score=" in reason:
                good_sim = float(reason.split("=")[1])

        # Use the actual API threshold (60%) to determine classification
        # The API returns bad_score as percentage (0-100)
        predicted_type = "spam" if phishing_score >= 60 else "legitimate"

        # Determine correctness
        if predicted_type == expected_type:
            correctness = "Correct"
        elif predicted_type == "spam" and expected_type == "legitimate":
            correctness = "FalsePositive"
        else:
            correctness = "FalseNegative"

        return {
            "EmailID": idx,
            "ExpectedType": expected_type,
            "PredictedType": predicted_type,
            "PhishingScore": phishing_score,
            "Confidence": confidence_level,
            "BadSim": bad_sim,
            "GoodSim": good_sim,
            "Correctness": correctness
        }
    except Exception as e:
        print(f"❌ Error analyzing email index={idx}: {e}")
        return {
            "EmailID": idx,
            "ExpectedType": expected_type,
            "PredictedType": "ERROR",
            "PhishingScore": None,
            "Confidence": "ERROR",
            "BadSim": None,
            "GoodSim": None,
            "Correctness": "AnalyzeException"
        }

# -------------------------------
# 🚀 Import (Train) Emails into API
# -------------------------------
print(f"📤 Importing {len(train_data)} training emails into the API...")

with ThreadPoolExecutor(max_workers=5) as executor, tqdm(total=len(train_data), desc="Importing", unit="emails") as pbar:
    futures = [executor.submit(process_email_for_insert, row) for _, row in train_data.iterrows()]
    for future in as_completed(futures):
        if future.result():
            pbar.update(1)

print("\n✅ Training data import completed!")

# -------------------------------
# 🛠️ Test (Analyze) the API
# -------------------------------
print(f"\n🔍 Testing {len(test_data)} emails...")
time.sleep(3)

analysis_rows = []
with tqdm(total=len(test_data), desc="Testing", unit="emails") as pbar:
    for idx, row in test_data.iterrows():
        result = analyze_email(row, idx)
        analysis_rows.append(result)
        pbar.update(1)

# -------------------------------
# 📊 Summarize & Write CSV
# -------------------------------
# Filter out any rows that had "AnalyzeFailed" or "AnalyzeException" if desired
tested_rows = [r for r in analysis_rows if r["Correctness"] not in ("AnalyzeFailed", "AnalyzeException", "ERROR")]

total_tested = len(tested_rows)
correct_classifications = sum(1 for r in tested_rows if r["Correctness"] == "Correct")
false_positives = sum(1 for r in tested_rows if r["Correctness"] == "FalsePositive")
false_negatives = sum(1 for r in tested_rows if r["Correctness"] == "FalseNegative")

# Confidence counts
high_confidence = sum(1 for r in tested_rows if r["Confidence"] == "High")
medium_confidence = sum(1 for r in tested_rows if r["Confidence"] == "Medium")
low_confidence = sum(1 for r in tested_rows if r["Confidence"] == "Low")

# Similarity scores
bad_sims = [r["BadSim"] for r in tested_rows if r["BadSim"] is not None]
good_sims = [r["GoodSim"] for r in tested_rows if r["GoodSim"] is not None]

# Calculate final metrics
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

    if bad_sims and good_sims:
        avg_bad = sum(bad_sims) / len(bad_sims)
        avg_good = sum(good_sims) / len(good_sims)
        print("\n🔎 Similarity Statistics:")
        print(f"Avg BadSim: {avg_bad:.3f}")
        print(f"Avg GoodSim: {avg_good:.3f}")
else:
    print("\n❌ No emails were tested or all failed analysis. Check dataset or API.")

# -------------------------------
# 📄 Write Detailed CSV
# -------------------------------
output_csv = "huggingface_test_results.csv"
fieldnames = [
    "EmailID",
    "ExpectedType",
    "PredictedType",
    "PhishingScore",
    "Confidence",
    "BadSim",
    "GoodSim",
    "Correctness"
]

with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in analysis_rows:
        writer.writerow(row)

print(f"\n✅ CSV results written to {output_csv}")
print("\n✅ Script Execution Completed!")
