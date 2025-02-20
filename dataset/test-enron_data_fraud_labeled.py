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
sample_size = 5000  # Number of random entries to test

# -------------------------------
# 🎯 Prepare Sample Data
# -------------------------------
# Randomly select rows, ensuring reproducibility with random_state=42
if len(data) < sample_size:
    print(f"⚠ The dataset has fewer than {sample_size} entries. Testing all available rows.")
    sample_data = data
else:
    sample_data = data.sample(n=sample_size, random_state=int(time.time()))

# Metrics counters
total_tested = 0
correct_classifications = 0
false_positives = 0  # predicted phishing, actually legitimate
false_negatives = 0  # predicted legitimate, actually phishing

# Count of phishing (spam) and legitimate (ham) emails
spam_count = 0  # Number of phishing emails tested
ham_count = 0   # Number of legitimate emails tested

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

        # If Body is empty or very short, skip or handle as needed
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

    print("\n📊 Test Summary:")
    print(f"Total Emails Tested: {total_tested}")
    print(f"Total Spam Emails Tested: {spam_count}")
    print(f"Total Ham Emails Tested: {ham_count}")
    print(f"Correct Classifications: {correct_classifications}")
    print(f"False Positives: {false_positives} ({false_positive_rate:.2f}%)")
    print(f"False Negatives: {false_negatives} ({false_negative_rate:.2f}%)")
    print(f"Overall Accuracy: {accuracy:.2f}%")
