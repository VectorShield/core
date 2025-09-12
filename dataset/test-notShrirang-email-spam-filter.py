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
# 📁 Dataset Path (Hugging Face)
# -------------------------------
# https://huggingface.co/datasets/NotShrirang/email-spam-filter
DATASET_PATH = "hf://datasets/NotShrirang/email-spam-filter/train.csv"

# -------------------------------
# ⚙️ API Endpoints
# -------------------------------
INSERT_API_URL = f"{os.getenv('API_URL', 'http://localhost:5000')}/insert"
ANALYZE_API_URL = f"{os.getenv('API_URL', 'http://localhost:5000')}/analyze"

# -------------------------------
# 📦 Load and Split Dataset
# -------------------------------
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# Ensure correct column names
df = df[['text', 'label_num']]
df.rename(columns={'text': 'Text', 'label_num': 'Spam'}, inplace=True)

# Separate spam and ham
spam_emails = df[df['Spam'] == 1]
ham_emails = df[df['Spam'] == 0]

# Define split ratio (80% train, 20% test)
train_spam = spam_emails.sample(frac=0.8, random_state=42)
test_spam = spam_emails.drop(train_spam.index)
train_ham = ham_emails.sample(frac=0.8, random_state=42)
test_ham = ham_emails.drop(train_ham.index)

# Combine train and test sets
train_data = pd.concat([train_spam, train_ham]).sample(frac=1, random_state=42)
test_data = pd.concat([test_spam, test_ham]).sample(frac=1, random_state=42)

print(f"✅ Dataset split successfully! (Train: {len(train_data)}, Test: {len(test_data)})")

# Count Spam and Ham in Test Set
spam_count = test_data[test_data["Spam"] == 1].shape[0]
ham_count = test_data[test_data["Spam"] == 0].shape[0]

print("📊 Test Dataset Breakdown:")
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
    """
    Extract a subject line if the first line starts with 'Subject:',
    otherwise treat all as body.
    """
    lines = text.split('\n')
    if lines and lines[0].startswith("Subject:"):
        subject = lines[0].replace("Subject:", "").strip()
        body = "\n".join(lines[1:]).strip()
    else:
        subject = ""
        body = text.strip()
    return subject, body

def insert_email(row):
    """
    Prepare and send email data to /insert.
    Return True if insertion succeeded, else False.
    """
    try:
        subject, body = parse_email_text(row["Text"])
        email_type = get_email_type(row["Spam"])

        payload = {
            "subject": subject,
            "body": base64.b64encode(body.encode("utf-8")).decode("utf-8"),
            "sender": "unknown@example.com",
            "type": email_type
        }

        resp = requests.post(INSERT_API_URL, json=payload)
        return resp.status_code == 200
    except Exception as e:
        print(f"❌ Exception inserting email: {e}")
        return False

def analyze_email(row, idx):
    """
    Sends the email to /analyze and collects:
      - EmailID (index)
      - ExpectedType
      - PredictedType
      - PhishingScore
      - Confidence
      - BadSim / GoodSim (parsed from 'reasons')
      - Correctness (Correct, FalsePositive, FalseNegative)
    Returns a dict for CSV export & stats.
    """
    subject, body = parse_email_text(row["Text"])
    expected_type = get_expected_classification(row["Spam"])  # "spam" or "legitimate"

    payload = {
        "subject": subject,
        "body": base64.b64encode(body.encode("utf-8")).decode("utf-8"),
        "sender": "unknown@example.com"
    }

    try:
        resp = requests.post(ANALYZE_API_URL, json=payload)
        if resp.status_code != 200:
            return {
                "EmailID": idx,
                "ExpectedType": expected_type,
                "PredictedType": "ERROR",
                "PhishingScore": None,
                "Confidence": "ERROR",
                "BadSim": None,
                "GoodSim": None,
                "Correctness": "AnalyzeFailed"
            }

        data = resp.json()
        phishing_score = data.get("phishing_score", 0)
        confidence_level = data.get("confidence_level", "Unknown")
        reasons = data.get("reasons", [])

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
        print(f"❌ Error analyzing email idx={idx}: {e}")
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
# 🚀 TRAIN (Insert) Emails
# -------------------------------
print(f"📤 Importing {len(train_data)} training emails into the API...")

with ThreadPoolExecutor(max_workers=5) as executor, tqdm(total=len(train_data), desc="Importing", unit="emails") as pbar:
    futures = [executor.submit(insert_email, row) for _, row in train_data.iterrows()]
    for fut in as_completed(futures):
        if fut.result():
            pbar.update(1)

print("\n✅ Training data import completed!")

# -------------------------------
# 🛠️ TEST (Analyze) the API
# -------------------------------
print(f"\n🔍 Testing {len(test_data)} emails...")
time.sleep(3)

analysis_rows = []
with tqdm(total=len(test_data), desc="Testing", unit="emails") as pbar:
    for idx, row in test_data.iterrows():
        result_dict = analyze_email(row, idx)
        analysis_rows.append(result_dict)
        pbar.update(1)

# -------------------------------
# 📊 Summarize + CSV
# -------------------------------
# Filter out rows with "AnalyzeFailed" / "AnalyzeException" if you only want valid results
valid_rows = [r for r in analysis_rows if r["Correctness"] not in ("AnalyzeFailed", "AnalyzeException")]

total_tested = len(valid_rows)
correct = sum(1 for r in valid_rows if r["Correctness"] == "Correct")
false_positives = sum(1 for r in valid_rows if r["Correctness"] == "FalsePositive")
false_negatives = sum(1 for r in valid_rows if r["Correctness"] == "FalseNegative")

if total_tested == 0:
    print("\n❌ No valid emails tested (or all failed). Check dataset or API.")
else:
    accuracy = (correct / total_tested) * 100
    fp_rate = (false_positives / total_tested) * 100
    fn_rate = (false_negatives / total_tested) * 100

    print("\n📊 Test Summary:")
    print(f"Total Emails Tested: {total_tested}")
    print(f"Correct Classifications: {correct} ({accuracy:.2f}%)")
    print(f"False Positives: {false_positives} ({fp_rate:.2f}%)")
    print(f"False Negatives: {false_negatives} ({fn_rate:.2f}%)")

# -------------------------------
# 📝 Write Detailed CSV
# -------------------------------
output_csv = "shrirang_spam_filter_results.csv"
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

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in analysis_rows:
        writer.writerow(row)

print(f"\n✅ CSV results written to {output_csv}")
print("\n✅ Script Execution Completed!")
