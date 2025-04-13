#!/usr/bin/env python3
import os
import csv
import re
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------
# ⚙️ API Endpoints
# -------------------------------
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
# 🔎 CSV + Analysis Storage
# -------------------------------
# We'll store each analysis result (one row per EML) here:
analysis_results = []

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
    payload = {
        "subject": email_data.get("subject", ""),
        "body": email_data.get("body", ""),
        "sender": email_data.get("sender", ""),
        "type": email_type,
    }
    resp = requests.post(INSERT_API_URL, json=payload)
    if resp.status_code != 200:
        print(f"❌ Insert failed for '{payload['subject']}': {resp.text}")
    else:
        print(f"Inserted [{email_type}] => {payload['subject']}")

def analyze_file(eml_data: dict, expected_type: str, filename: str):
    """
    Calls /analyze to check classification. Returns a dict with the info needed for CSV/report.
    
    - filename
    - confidence_level (High/Medium/Low)
    - phish_sim
    - legit_sim
    - predicted_type ("phishing" or "legitimate")
    - expected_type
    - correctness label ("Correct", "FalsePositive", or "FalseNegative")
    """
    payload = {
        "subject": eml_data.get("subject", ""),
        "body": eml_data.get("body", ""),
        "sender": eml_data.get("sender", ""),
    }

    resp = requests.post(ANALYZE_API_URL, json=payload)
    if resp.status_code != 200:
        # Return a row with minimal info so we can track the error
        return {
            "filename": filename,
            "confidence": "ERROR",
            "PhishSim": None,
            "LegitSim": None,
            "ScanResult": "ERROR",
            "ExpectedResult": expected_type,
            "MisclassificationType": "AnalyzeFailed",
        }

    data = resp.json()
    phishing_score = data.get("phishing_score", 0)
    confidence_level = data.get("confidence_level", "Unknown")
    reasons = data.get("reasons", [])

    # Attempt to parse "PhishSim=xx, LegitSim=yy" from reasons
    phish_sim = None
    legit_sim = None
    for reason in reasons:
        match = re.search(r"PhishSim=([\d.]+).*LegitSim=([\d.]+)", reason)
        if match:
            phish_sim = float(match.group(1))
            legit_sim = float(match.group(2))
            break

    # We'll label predicted as phishing if phishing_score >= 70
    predicted_type = "spam" if phishing_score >= 70 else "business"


    # Determine misclassification type
    if predicted_type == expected_type:
        misclass_type = "Correct"
    elif predicted_type == "phishing" and expected_type == "legitimate":
        misclass_type = "FalsePositive"
    else:
        misclass_type = "FalseNegative"

    return {
        "filename": filename,
        "confidence": confidence_level,
        "PhishSim": phish_sim,
        "LegitSim": legit_sim,
        "ScanResult": predicted_type,
        "ExpectedResult": expected_type,
        "MisclassificationType": misclass_type,
    }

def process_folder(folder_path: str, label: str, do_insert=True, do_analyze=False, max_analyze_workers=4):
    """
    1) For each EML file in folder_path:
       - parse via /parse_eml
       - insert via /insert (if do_insert=True)
       - analyze via /analyze (if do_analyze=True), done in parallel
    2) label is 'phishing' or 'legitimate'
    """
    if not os.path.isdir(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    # Collect .eml files
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".eml")]
    print(f"Found {len(files)} EML files in {folder_path}")

    # We'll parse + optionally insert sequentially, then do parallel analyze
    analyze_tasks = []

    for fname in files:
        fpath = os.path.join(folder_path, fname)
        eml_data = parse_eml_file(fpath)
        if not eml_data:
            continue  # skip if parse failed

        # Insert step
        if do_insert:
            insert_email(eml_data, label)

        # Prepare for analyze
        if do_analyze:
            analyze_tasks.append((eml_data, label, fname))

    # Now we do concurrency for analyzing
    if do_analyze and analyze_tasks:
        with ThreadPoolExecutor(max_workers=max_analyze_workers) as executor:
            futures = [executor.submit(analyze_file, eml, lbl, fn) for (eml, lbl, fn) in analyze_tasks]
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    analysis_results.append(result)

def write_results_to_csv(rows, output_file="results.csv"):
    """
    Writes the analysis rows to a CSV file with the desired columns.
    """
    fieldnames = [
        "filename",
        "confidence",
        "PhishSim",
        "LegitSim",
        "ScanResult",
        "ExpectedResult",
        "MisclassificationType"
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Analysis results written to {output_file}")

def main():
    # -------------------------------
    # 🚀 TRAIN STEP (Insert)
    # -------------------------------
    print("=== TRAIN PHASE: Importing EML files into the vector store ===")
    # Insert spam as 'phishing'
    process_folder(SPAM_FOLDER, label="spam", do_insert=True, do_analyze=False)
    # Insert ham as 'legitimate'
    process_folder(HAM_FOLDER,  label="business", do_insert=True, do_analyze=False)

    # Wait a bit so batch upsert can finish (especially if there's a background queue)
    print("Waiting 10 seconds for background upserts...")
    time.sleep(10)

    # -------------------------------
    # 🛠️ TEST STEP (Analyze)
    # -------------------------------
    print("\n=== TEST PHASE: Checking model predictions on EML files ===")
    process_folder(SPAM_FOLDER, label="phishing", do_insert=False, do_analyze=True, max_analyze_workers=4)
    process_folder(HAM_FOLDER,  label="legitimate", do_insert=False, do_analyze=True, max_analyze_workers=4)

    if not analysis_results:
        print("\n❌ No emails analyzed. Check your EML directories.")
        return

    # -------------------------------
    # 📊 Summarize + CSV
    # -------------------------------
    # 1) Write the results to CSV
    write_results_to_csv(analysis_results, output_file="results.csv")

    # 2) Calculate metrics
    total_tested = len(analysis_results)
    correct = sum(1 for r in analysis_results if r["MisclassificationType"] == "Correct")
    false_pos = sum(1 for r in analysis_results if r["MisclassificationType"] == "FalsePositive")
    false_neg = sum(1 for r in analysis_results if r["MisclassificationType"] == "FalseNegative")

    # Confidence stats
    high_conf = sum(1 for r in analysis_results if r["confidence"] == "High")
    med_conf  = sum(1 for r in analysis_results if r["confidence"] == "Medium")
    low_conf  = sum(1 for r in analysis_results if r["confidence"] == "Low")

    accuracy = (correct / total_tested) * 100 if total_tested else 0
    fp_rate  = (false_pos / total_tested) * 100 if total_tested else 0
    fn_rate  = (false_neg / total_tested) * 100 if total_tested else 0
    high_conf_rate = (high_conf / total_tested) * 100 if total_tested else 0
    med_conf_rate  = (med_conf / total_tested) * 100 if total_tested else 0
    low_conf_rate  = (low_conf / total_tested) * 100 if total_tested else 0

    print("\n=== FINAL TEST REPORT ===")
    print(f"Total EMLs Tested:           {total_tested}")
    print(f"Correct Classifications:     {correct} ({accuracy:.2f}%)")
    print(f"False Positives:            {false_pos} ({fp_rate:.2f}%)")
    print(f"False Negatives:            {false_neg} ({fn_rate:.2f}%)")

    print("\nConfidence-Level Breakdown:")
    print(f"  - High Confidence:   {high_conf} ({high_conf_rate:.2f}%)")
    print(f"  - Medium Confidence: {med_conf} ({med_conf_rate:.2f}%)")
    print(f"  - Low Confidence:    {low_conf} ({low_conf_rate:.2f}%)")

if __name__ == "__main__":
    main()
