#!/usr/bin/env python3
import os
import requests
import time
import math

# -------------------------------
# ⚙️ API Endpoints
# -------------------------------
# Adjust to match your actual host/port if not localhost:5000
PARSE_EML_API_URL = os.getenv("PARSE_EML_API_URL", "http://localhost:5000/parse_eml")
INSERT_API_URL    = os.getenv("INSERT_API_URL",    "http://localhost:5000/insert")
ANALYZE_API_URL   = os.getenv("ANALYZE_API_URL",   "http://localhost:5000/analyze")

# -------------------------------
# 📁 Folder Paths
# -------------------------------
BASE_PATH = "./kaltenecker.m"
SPAM_FOLDER = os.path.join(BASE_PATH, "spam")
HAM_FOLDER  = os.path.join(BASE_PATH, "ham")

# -------------------------------
# 📊 Metrics Tracking
# -------------------------------
total_tested = 0
correct_classifications = 0
false_positives = 0   # Predicted phishing, actually ham
false_negatives = 0   # Predicted ham, actually spam

high_confidence = 0
medium_confidence = 0
low_confidence = 0

def parse_eml_file(file_path: str):
    """
    Calls the /parse_eml endpoint to parse raw EML into subject, body (base64), and sender.
    Returns a dict: { "subject": ..., "body": ..., "sender": ... }
    """
    with open(file_path, "rb") as f:
        resp = requests.post(PARSE_EML_API_URL, files={"file": f})
    if resp.status_code != 200:
        print(f"❌ Could not parse EML '{file_path}': {resp.text}")
        return None

    data = resp.json()
    # data should look like: {"message": "Parsed EML", "email": {...}}
    email_obj = data.get("email")
    if not email_obj:
        print(f"❌ Invalid parse response for '{file_path}': {resp.text}")
        return None

    return email_obj

def insert_email(email_data: dict, email_type: str):
    """
    Calls /insert to store parsed email in Qdrant with label ('phishing' or 'legitimate').
    email_data should have 'subject', 'body', 'sender'.
    """
    # The EmailRequest also expects optional 'reply_to', 'attachments', 'customerId', etc.
    payload = {
        "subject": email_data.get("subject", ""),
        "body": email_data.get("body", ""),
        "sender": email_data.get("sender", ""),
        "type": email_type
    }
    resp = requests.post(INSERT_API_URL, json=payload)
    if resp.status_code != 200:
        print(f"❌ Insert failed for '{payload['subject']}': {resp.text}")
    else:
        # Just log the success message
        print(f"Inserted [{email_type}] => {payload['subject']}")

def analyze_email(email_data: dict, expected_type: str):
    """
    Calls /analyze to check classification. Updates global metrics.
    expected_type is 'phishing' or 'legitimate'.
    """
    global total_tested, correct_classifications
    global false_positives, false_negatives
    global high_confidence, medium_confidence, low_confidence

    payload = {
        "subject": email_data.get("subject", ""),
        "body": email_data.get("body", ""),
        "sender": email_data.get("sender", ""),
    }
    resp = requests.post(ANALYZE_API_URL, json=payload)
    if resp.status_code != 200:
        print(f"❌ Analyze failed for '{payload['subject']}': {resp.text}")
        return

    data = resp.json()
    phishing_score   = data.get("phishing_score", 0)
    confidence_level = data.get("confidence_level", "Unknown")

    # We'll label predicted as phishing if phishing_score >= 70 (adjust threshold as you wish)
    predicted_type = "phishing" if phishing_score >= 70 else "legitimate"

    # Update confidence-level counters
    if confidence_level == "High":
        high_confidence += 1
    elif confidence_level == "Medium":
        medium_confidence += 1
    elif confidence_level == "Low":
        low_confidence += 1

    # Compare predicted to expected
    total_tested += 1
    if predicted_type == expected_type:
        correct_classifications += 1
    else:
        # If mismatch, see which type of error
        if predicted_type == "phishing" and expected_type == "legitimate":
            false_positives += 1
        elif predicted_type == "legitimate" and expected_type == "phishing":
            false_negatives += 1

def process_folder(folder_path: str, label: str, do_insert=True, do_analyze=False):
    """
    1) For each EML file in folder_path,
       - parse with /parse_eml
       - optionally call /insert or /analyze
    2) label is 'phishing' or 'legitimate'
    """
    if not os.path.isdir(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    # Collect .eml files
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".eml")]
    print(f"Found {len(files)} EML files in {folder_path}")

    for fname in files:
        fpath = os.path.join(folder_path, fname)
        eml_data = parse_eml_file(fpath)
        if not eml_data:
            continue  # skip if parse failed

        if do_insert:
            insert_email(eml_data, label)
        if do_analyze:
            analyze_email(eml_data, label)

def main():
    # -------------------------------
    # 🚀 TRAIN STEP (Insert)
    # -------------------------------
    print("=== TRAIN PHASE: Importing EML files into the vector store ===")
    # Insert spam as 'phishing'
    process_folder(SPAM_FOLDER, label="phishing", do_insert=True, do_analyze=False)
    # Insert ham as 'legitimate'
    process_folder(HAM_FOLDER,  label="legitimate", do_insert=True, do_analyze=False)

    # Wait a bit so batch upsert can finish (especially if there's a background queue)
    print("Waiting 10 seconds for background upserts...")
    time.sleep(10)

    # -------------------------------
    # 🛠️ TEST STEP (Analyze)
    # -------------------------------
    print("\n=== TEST PHASE: Checking model predictions on EML files ===")
    process_folder(SPAM_FOLDER, label="phishing", do_insert=False, do_analyze=True)
    process_folder(HAM_FOLDER,  label="legitimate", do_insert=False, do_analyze=True)

    # -------------------------------
    # 📊 Print Final Report
    # -------------------------------
    if total_tested == 0:
        print("\n❌ No emails tested. Check your EML directories.")
        return

    accuracy = (correct_classifications / total_tested) * 100
    fp_rate  = (false_positives / total_tested) * 100
    fn_rate  = (false_negatives / total_tested) * 100
    high_conf_rate   = (high_confidence / total_tested) * 100
    medium_conf_rate = (medium_confidence / total_tested) * 100
    low_conf_rate    = (low_confidence / total_tested) * 100

    print("\n=== FINAL TEST REPORT ===")
    print(f"Total EMLs Tested:           {total_tested}")
    print(f"Correct Classifications:     {correct_classifications} ({accuracy:.2f}%)")
    print(f"False Positives:            {false_positives} ({fp_rate:.2f}%)")
    print(f"False Negatives:            {false_negatives} ({fn_rate:.2f}%)")

    print("\nConfidence-Level Breakdown:")
    print(f"  - High Confidence:   {high_confidence} ({high_conf_rate:.2f}%)")
    print(f"  - Medium Confidence: {medium_confidence} ({medium_conf_rate:.2f}%)")
    print(f"  - Low Confidence:    {low_confidence} ({low_conf_rate:.2f}%)")

if __name__ == "__main__":
    main()
