import os
import pandas as pd
import requests
import base64
import time

# -------------------------------
# 📁 Setup File Paths
# -------------------------------
current_dir = os.path.dirname(os.path.abspath("__file__"))
csv_file_path = os.path.join(current_dir, "dataset", "enron_data_fraud_labeled.csv")

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
    """
    Determines the email classification ('phishing' or 'legitimate')
    based on the 'Label' field from the dataset.
    - Assumes 'Label' == 1 => 'phishing'
    -          'Label' == 0 => 'legitimate'
    """
    return "phishing" if label == 1 else "legitimate"

# -------------------------------
# 🚀 Test Script Configuration
# -------------------------------
analyze_api_url = "http://localhost:5000/analyze"
sample_size = 1000  # Total number of random entries to test

# -------------------------------
# 🎯 Prepare Sample Data
# -------------------------------
# Separate spam (phishing) and ham (legitimate)
spam_data = data[data["Label"] == 1]
ham_data = data[data["Label"] == 0]

# Define the desired ratio
spam_ratio = 0.3  # 30% spam
ham_ratio = 0.7   # 70% ham

# Calculate how many spam and ham samples to pick
spam_sample_size = int(sample_size * spam_ratio)
ham_sample_size = sample_size - spam_sample_size

# Check if we have enough spam/ham emails
if len(spam_data) < spam_sample_size or len(ham_data) < ham_sample_size:
    print("⚠ Not enough spam or ham to meet the desired ratio.")
    # In this fallback, sample everything we can while respecting the ratio if possible
    spam_sample_size = min(len(spam_data), spam_sample_size)
    ham_sample_size = min(len(ham_data), ham_sample_size)

# Now sample spam and ham separately
spam_sample = spam_data.sample(n=spam_sample_size, random_state=int(time.time()))
ham_sample = ham_data.sample(n=ham_sample_size, random_state=int(time.time()))

# Combine them into a single DataFrame
sample_data = pd.concat([spam_sample, ham_sample]).sample(frac=1, random_state=int(time.time()))

if len(sample_data) == 0:
    print("⚠ No data available for testing after applying ratio-based sampling.")
    exit(1)

# -------------------------------
# 🔍 Initialize Metrics
# -------------------------------
total_tested = 0
correct_classifications = 0
false_positives = 0  # predicted phishing, actually legitimate
false_negatives = 0  # predicted legitimate, actually phishing

spam_count = 0  # Number of phishing emails tested
ham_count = 0   # Number of legitimate emails tested

high_confidence = 0
medium_confidence = 0
low_confidence = 0

# -------------------------------
# 🔍 Validate Each Email
# -------------------------------
for index, row in sample_data.iterrows():
    try:
        # Extract fields, handling missing data
        email_body_raw = row["Body"] if pd.notna(row["Body"]) else ""
        email_subject = row["Subject"] if pd.notna(row["Subject"]) else "No Subject"
        email_sender = row["From"] if pd.notna(row["From"]) else "unknown@enron.com"
        label_raw = row["Label"] if pd.notna(row["Label"]) else 0  # fallback to 0 if missing
        expected_type = get_email_type(label_raw)

        # Count email type occurrences
        if expected_type == "phishing":
            spam_count += 1
        else:
            ham_count += 1

        # If Body is empty or very short, handle or skip
        if not email_body_raw.strip():
            pass  # Optionally skip or continue

        # Base64-encode the body for the request
        encoded_body = base64.b64encode(email_body_raw.encode("utf-8")).decode("utf-8")

        # Prepare the request payload
        payload = {
            "subject": email_subject,
            "body": encoded_body,
            "sender": email_sender
        }

        # Send the POST request to the analyze endpoint
        response = requests.post(analyze_api_url, json=payload)
        response_data = response.json()

        if response.status_code == 200:
            total_tested += 1
            # Extract confidence level
            confidence_level = response_data.get("confidence_level", "Unknown")

            # Count confidence levels
            if confidence_level == "High":
                high_confidence += 1
            elif confidence_level == "Medium":
                medium_confidence += 1
            elif confidence_level == "Low":
                low_confidence += 1

            # Simple thresholding logic: phishing_score >= 70 => "phishing"
            predicted_type = "phishing" if response_data["phishing_score"] >= 70 else "legitimate"

            if predicted_type == expected_type:
                correct_classifications += 1
            elif predicted_type == "phishing" and expected_type == "legitimate":
                false_positives += 1
            elif predicted_type == "legitimate" and expected_type == "phishing":
                false_negatives += 1
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
