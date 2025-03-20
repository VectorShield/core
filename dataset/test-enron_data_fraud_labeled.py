import os
import re
import time
import base64
import requests
import pandas as pd
from tqdm import tqdm

# -------------------------------
# 📁 Setup File Paths
# -------------------------------
current_dir = os.path.dirname(os.path.abspath("__file__"))
csv_file_path = os.path.join(current_dir, "dataset", "test_data-enron_data_fraud_labeled.csv")

# -------------------------------
# 📦 Load Dataset
# -------------------------------
try:
    data = pd.read_csv(csv_file_path)
    print(f"Dataset loaded successfully from: {csv_file_path}")
except FileNotFoundError:
    print(f"❌ CSV file not found at: {csv_file_path}")
    exit(1)

# -------------------------------
# 📌 Helper Functions
# -------------------------------
def get_email_type(label):
    return "phishing" if label == 1 else "legitimate"

# -------------------------------
# 🚀 Test Script Configuration
# -------------------------------
analyze_api_url = "http://localhost:5000/analyze"
sample_size = 200  # Total number of random entries to test

# -------------------------------
# 🎯 Prepare Sample Data
# -------------------------------
spam_data = data[data["Label"] == 1]
ham_data = data[data["Label"] == 0]

spam_ratio = 0.3
ham_ratio = 0.7

spam_sample_size = int(sample_size * spam_ratio)
ham_sample_size = sample_size - spam_sample_size

if len(spam_data) < spam_sample_size or len(ham_data) < ham_sample_size:
    print("⚠ Not enough spam or ham to meet the desired ratio.")
    spam_sample_size = min(len(spam_data), spam_sample_size)
    ham_sample_size = min(len(ham_data), ham_sample_size)

spam_sample = spam_data.sample(n=spam_sample_size, random_state=int(time.time()))
ham_sample = ham_data.sample(n=ham_sample_size, random_state=int(time.time()))

sample_data = pd.concat([spam_sample, ham_sample]).sample(frac=1, random_state=int(time.time()))

if len(sample_data) == 0:
    print("⚠ No data available for testing after applying ratio-based sampling.")
    exit(1)

# -------------------------------
# 🔍 Initialize Metrics
# -------------------------------
total_tested = 0
correct_classifications = 0
false_positives = 0
false_negatives = 0

spam_count = 0
ham_count = 0

high_confidence = 0
medium_confidence = 0
low_confidence = 0

phish_sims = []
legit_sims = []

# -------------------------------
# 🔍 Validate Each Email
# -------------------------------
for index, row in tqdm(sample_data.iterrows(), total=len(sample_data), desc="Analyzing", unit="emails"):
    try:
        email_body_raw = row["Body"] if pd.notna(row["Body"]) else ""
        email_subject = row["Subject"] if pd.notna(row["Subject"]) else "No Subject"
        email_sender = row["From"] if pd.notna(row["From"]) else "unknown@enron.com"
        label_raw = row["Label"] if pd.notna(row["Label"]) else 0
        expected_type = get_email_type(label_raw)

        if expected_type == "phishing":
            spam_count += 1
        else:
            ham_count += 1

        encoded_body = base64.b64encode(email_body_raw.encode("utf-8")).decode("utf-8")
        payload = {
            "subject": email_subject,
            "body": encoded_body,
            "sender": email_sender
        }

        response = requests.post(analyze_api_url, json=payload)
        response_data = response.json()

        if response.status_code == 200:
            total_tested += 1
            confidence_level = response_data.get("confidence_level", "Unknown")

            if confidence_level == "High":
                high_confidence += 1
            elif confidence_level == "Medium":
                medium_confidence += 1
            elif confidence_level == "Low":
                low_confidence += 1

            predicted_type = "phishing" if response_data["phishing_score"] >= 70 else "legitimate"

            if predicted_type == expected_type:
                correct_classifications += 1
            elif predicted_type == "phishing" and expected_type == "legitimate":
                false_positives += 1
            elif predicted_type == "legitimate" and expected_type == "phishing":
                false_negatives += 1

            # Parse "PhishSim=..., LegitSim=..." from reasons (if present)
            reasons_list = response_data.get("reasons", [])
            for reason in reasons_list:
                # Example reason: "PhishSim=0.34, LegitSim=0.02"
                match = re.search(r"PhishSim=([\d.]+).+LegitSim=([\d.]+)", reason)
                if match:
                    ph_val = float(match.group(1))
                    lg_val = float(match.group(2))
                    phish_sims.append(ph_val)
                    legit_sims.append(lg_val)

        else:
            print(f"❌ Error analyzing row {index}: status {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ An error occurred at row {index}: {e}")

# -------------------------------
# 📊 Generate Final Report
# -------------------------------
if total_tested == 0:
    print("\nNo emails were tested. Something might be wrong with the dataset or requests.")
else:
    accuracy = (correct_classifications / total_tested) * 100
    false_positive_rate = (false_positives / total_tested) * 100
    false_negative_rate = (false_negatives / total_tested) * 100

    high_confidence_rate = (high_confidence / total_tested) * 100
    medium_confidence_rate = (medium_confidence / total_tested) * 100
    low_confidence_rate = (low_confidence / total_tested) * 100

    print("\n📊 Test Summary:")
    print(f"Total Emails Tested: {total_tested}")
    print(f"Total Spam Emails Tested: {spam_count}")
    print(f"Total Ham Emails Tested: {ham_count}")
    print(f"Correct Classifications: {correct_classifications}")
    print(f"False Positives: {false_positives} ({false_positive_rate:.2f}%)")
    print(f"False Negatives: {false_negatives} ({false_negative_rate:.2f}%)")
    print(f"Overall Accuracy: {accuracy:.2f}%")

    print("\n🔍 Confidence Level Statistics:")
    print(f"High Confidence: {high_confidence} ({high_confidence_rate:.2f}%)")
    print(f"Medium Confidence: {medium_confidence} ({medium_confidence_rate:.2f}%)")
    print(f"Low Confidence: {low_confidence} ({low_confidence_rate:.2f}%)")

    # Additional stats about the "PhishSim" vs. "LegitSim" values
    if phish_sims and legit_sims:
        avg_phish = sum(phish_sims) / len(phish_sims)
        avg_legit = sum(legit_sims) / len(legit_sims)
        print("\n🔎 Similarity Statistics (from reasons):")
        print(f"Average PhishSim: {avg_phish:.3f}")
        print(f"Average LegitSim: {avg_legit:.3f}")
        print(f"Data points counted: {len(phish_sims)}")
    else:
        print("\n🔎 No PhishSim/LegitSim data found in reasons.")
