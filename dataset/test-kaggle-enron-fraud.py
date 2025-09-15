#!/usr/bin/env python3
"""
Kaggle Enron Fraud Email Dataset Tester

This script downloads and tests the Enron Fraud Email Dataset from Kaggle using kagglehub.
It follows the same pattern as other test scripts: import training data, then analyze test data.

Dataset: https://www.kaggle.com/datasets/advaithsrao/enron-fraud-email-dataset

Usage:
    pip install kagglehub
    python dataset/test-kaggle-enron-fraud.py

The script will:
1. Download the Enron fraud dataset using kagglehub
2. Load and analyze the dataset structure
3. Import training emails into the vector database via /insert
4. Test the model against validation emails via /analyze
5. Generate detailed performance reports and CSV output
"""

import os
import time
import base64
import requests
import pandas as pd
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------
# ⚙️ API Endpoints
# -------------------------------
INSERT_API_URL = f"{os.getenv('API_URL', 'http://localhost:5000')}/insert"
ANALYZE_API_URL = f"{os.getenv('API_URL', 'http://localhost:5000')}/analyze"

# -------------------------------
# 📦 Download and Load Dataset
# -------------------------------
print("📥 Downloading Enron Fraud Email Dataset from Kaggle...")

try:
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("advaithsrao/enron-fraud-email-dataset")
    print(f"✅ Dataset downloaded to: {path}")

    # Load the CSV file
    csv_file = os.path.join(path, "enron_data_fraud_labeled.csv")
    if not os.path.exists(csv_file):
        # Try alternative filename
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if csv_files:
            csv_file = os.path.join(path, csv_files[0])
            print(f"📋 Using CSV file: {csv_files[0]}")
        else:
            raise FileNotFoundError("No CSV file found in dataset")

    data = pd.read_csv(csv_file)
    print(f"✅ Dataset loaded successfully! Shape: {data.shape}")

except ImportError:
    print("❌ kagglehub not installed. Please run: pip install kagglehub")
    exit(1)
except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    exit(1)

# -------------------------------
# 📊 Analyze Dataset Structure
# -------------------------------
print("\n📊 Dataset Analysis:")
print(f"Columns: {list(data.columns)}")
print(f"Shape: {data.shape}")

# Try to identify the relevant columns for email analysis
email_columns = []
label_column = None

# Common patterns for email datasets
possible_email_cols = ['text', 'body', 'content', 'message', 'email', 'subject']
possible_label_cols = ['label', 'class', 'target', 'fraud', 'poi', 'spam', 'type']

for col in data.columns:
    col_lower = col.lower()
    if any(pattern in col_lower for pattern in possible_email_cols):
        email_columns.append(col)
    if any(pattern in col_lower for pattern in possible_label_cols):
        label_column = col

print(f"Potential email columns: {email_columns}")
print(f"Potential label column: {label_column}")

# Show first few rows
print(f"\nFirst 3 rows:")
print(data.head(3))

# Show value counts for label column if found
if label_column:
    print(f"\nLabel distribution ({label_column}):")
    print(data[label_column].value_counts())

# -------------------------------
# 🔧 Data Preprocessing
# -------------------------------
# If we can't automatically detect columns, let user know
if not email_columns or not label_column:
    print("\n⚠️  Could not automatically detect email and label columns.")
    print("Please examine the dataset structure above and modify the script accordingly.")

    # Try some common fallbacks
    if 'message' in data.columns and 'poi' in data.columns:
        email_columns = ['message']
        label_column = 'poi'
        print(f"🔄 Using fallback: email='{email_columns[0]}', label='{label_column}'")
    elif len(data.columns) >= 2:
        # Use the last column as label, second-to-last as email content
        email_columns = [data.columns[-2]]
        label_column = data.columns[-1]
        print(f"🔄 Using heuristic: email='{email_columns[0]}', label='{label_column}'")
    else:
        print("❌ Cannot proceed without identifying email content and labels.")
        exit(1)

# Use the first email column
email_col = email_columns[0]

# Clean and prepare data
print(f"\n🧹 Preprocessing data...")
data = data.dropna(subset=[email_col, label_column])
print(f"After removing NaN values: {data.shape}")

# Convert labels to binary if needed
label_counts = data[label_column].value_counts()
print(f"Label distribution: {label_counts}")

# -------------------------------
# 📌 Helper Functions
# -------------------------------
def get_email_type(label_value):
    """Convert label to email type for API"""
    # Handle different label formats
    if isinstance(label_value, str):
        label_lower = label_value.lower()
        if any(word in label_lower for word in ['fraud', 'spam', 'phish', 'poi', '1', 'true']):
            return "spam"
        else:
            return "business"
    elif isinstance(label_value, (int, float)):
        return "spam" if label_value > 0 else "business"
    else:
        return "business"

def get_expected_classification(label_value):
    """Convert label to expected classification for comparison"""
    # Handle different label formats
    if isinstance(label_value, str):
        label_lower = label_value.lower()
        if any(word in label_lower for word in ['fraud', 'spam', 'phish', 'poi', '1', 'true']):
            return "spam"
        else:
            return "legitimate"
    elif isinstance(label_value, (int, float)):
        return "spam" if label_value > 0 else "legitimate"
    else:
        return "legitimate"

def parse_email_text(text):
    """Parse email content - treat entire text as body with empty subject"""
    if pd.isna(text) or not str(text).strip():
        return "", "No content available"
    return "", str(text).strip()

def process_email_for_insert(row):
    """Prepare and send email data to /insert endpoint"""
    try:
        subject, body = parse_email_text(row[email_col])
        email_type = get_email_type(row[label_column])

        payload = {
            "subject": subject,
            "body": base64.b64encode(body.encode("utf-8")).decode("utf-8"),
            "sender": "enron@example.com",
            "type": email_type
        }

        resp = requests.post(INSERT_API_URL, json=payload)
        return (resp.status_code == 200)
    except Exception as e:
        print(f"❌ Exception inserting email: {e}")
        return False

def analyze_email(row, idx):
    """Send email to /analyze and gather results for CSV"""
    subject, body = parse_email_text(row[email_col])
    expected_type = get_expected_classification(row[label_column])

    payload = {
        "subject": subject,
        "body": base64.b64encode(body.encode("utf-8")).decode("utf-8"),
        "sender": "enron@example.com"
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

        # Extract similarity scores from reasons
        bad_sim = None
        good_sim = None
        for reason in reasons:
            if "weighted_bad_score=" in reason:
                bad_sim = float(reason.split("=")[1])
            elif "weighted_good_score=" in reason:
                good_sim = float(reason.split("=")[1])

        # Use 60% threshold for classification
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
# 📂 Split Data
# -------------------------------
# Use 70% for training, 30% for testing
train_size = int(0.7 * len(data))
train_data = data.iloc[:train_size].copy()
test_data = data.iloc[train_size:].copy()

print(f"\n📊 Data Split:")
print(f"Training set: {len(train_data)} emails")
print(f"Test set: {len(test_data)} emails")

# Count fraud/legitimate in each set
train_fraud = sum(get_email_type(label) == "spam" for label in train_data[label_column])
train_legit = len(train_data) - train_fraud
test_fraud = sum(get_expected_classification(label) == "spam" for label in test_data[label_column])
test_legit = len(test_data) - test_fraud

print(f"Training: {train_fraud} fraud, {train_legit} legitimate")
print(f"Test: {test_fraud} fraud, {test_legit} legitimate")

# -------------------------------
# 🚀 Import Training Data
# -------------------------------
print(f"\n📤 Importing {len(train_data)} training emails into the API...")

with ThreadPoolExecutor(max_workers=5) as executor, tqdm(total=len(train_data), desc="Importing", unit="emails") as pbar:
    futures = [executor.submit(process_email_for_insert, row) for _, row in train_data.iterrows()]
    successful_imports = 0
    for future in as_completed(futures):
        if future.result():
            successful_imports += 1
        pbar.update(1)

print(f"\n✅ Training data import completed! ({successful_imports}/{len(train_data)} successful)")

# -------------------------------
# 🛠️ Test the API
# -------------------------------
print(f"\n🔍 Testing {len(test_data)} emails...")
time.sleep(3)

analysis_rows = []
with tqdm(total=len(test_data), desc="Testing", unit="emails") as pbar:
    for idx, (original_idx, row) in enumerate(test_data.iterrows()):
        result = analyze_email(row, original_idx)
        analysis_rows.append(result)
        pbar.update(1)

# -------------------------------
# 📊 Calculate Metrics
# -------------------------------
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
output_csv = "kaggle_enron_fraud_results.csv"
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
print(f"📄 Dataset source: {csv_file}")
print("\n✅ Script Execution Completed!")