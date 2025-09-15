#!/usr/bin/env python3
"""
EML Dataset Tester

This script downloads, imports, and tests classified EML email datasets for the VectorShield application.
It follows the same pattern as other test scripts: import training data, then analyze test data.

Usage:
    python dataset/test-eml-datasets.py

The script will:
1. Download and organize EML datasets from various sources
2. Parse EML files using the /parse_eml API endpoint
3. Import training emails into the vector database via /insert
4. Test the model against validation emails via /analyze
5. Generate detailed performance reports and CSV output
"""

import os
import requests
import csv
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# -------------------------------
# ⚙️ Configuration
# -------------------------------
API_BASE_URL = os.getenv("API_URL", "http://localhost:5000")
PARSE_EML_API_URL = f"{API_BASE_URL}/parse_eml"
INSERT_API_URL = f"{API_BASE_URL}/insert"
ANALYZE_API_URL = f"{API_BASE_URL}/analyze"

# Dataset configuration
DATASET_DIR = "eml_datasets"
DOWNLOAD_DATE = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Sample datasets (GitHub repositories with EML files)
DATASETS = {
    "phishing_pot": {
        "name": "Phishing Pot Collection",
        "repo": "rf-peixoto/phishing_pot",
        "type": "spam",
        "description": "Real phishing samples collected via honey pots"
    },
    "bayes_spam_filter": {
        "name": "Bayes Spam Filter Test Data",
        "repo": "bierik/bayes-spam-filter",
        "path": "src/test/resources",
        "type": "mixed",
        "description": "Spam and ham EML files for naive bayes testing"
    },
    "email_analyzer": {
        "name": "Email Analyzer Samples",
        "repo": "MrCalv1n/EmailAnalyzer",
        "path": "samples",
        "type": "mixed",
        "description": "Sample EML files for email analysis"
    },
    "manual_samples": {
        "name": "Manual Sample Collection",
        "repo": None,
        "type": "manual",
        "description": "Manually create sample EML files if no repos have sufficient data"
    }
}

# Test results storage
analysis_results = []

# -------------------------------
# 📁 Helper Functions
# -------------------------------
def create_directories():
    """Create necessary directory structure"""
    base_path = Path(DATASET_DIR)
    subdirs = ["spam", "legitimate", "mixed", "metadata", "downloads"]

    for subdir in subdirs:
        (base_path / subdir).mkdir(parents=True, exist_ok=True)

    print(f"✅ Created directory structure in {DATASET_DIR}/")

def download_file(url: str, filename: str) -> bool:
    """Download a file from URL"""
    try:
        print(f"📥 Downloading {filename}...")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def download_github_file(repo: str, file_path: str, local_path: str) -> bool:
    """Download a single file from GitHub repository"""
    # Try main branch first, then master branch
    for branch in ["main", "master"]:
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file_path}"
        if download_file(url, local_path):
            return True
    return False

def fetch_github_files(repo: str, path: str = "") -> list:
    """Fetch list of files from GitHub repository"""
    eml_files = []

    # Try both main and master branches
    for branch in ["main", "master"]:
        try:
            url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            files = response.json()

            # Handle both single files and directories
            if isinstance(files, dict):
                files = [files]

            for file_info in files:
                if file_info['name'].endswith('.eml'):
                    eml_files.append({
                        'name': file_info['name'],
                        'download_url': file_info['download_url'],
                        'path': file_info['path'],
                        'branch': branch
                    })
                elif file_info['type'] == 'dir':
                    # Recursively fetch from subdirectories
                    subdir_files = fetch_github_files(repo, file_info['path'])
                    eml_files.extend(subdir_files)

            if eml_files:  # If we found files, no need to try other branch
                break

        except Exception as e:
            print(f"⚠️  Failed to fetch from {repo}/{branch}: {e}")
            continue

    return eml_files

def parse_eml_file(file_path: str) -> dict:
    """Parse EML file using the API"""
    try:
        with open(file_path, "rb") as f:
            response = requests.post(PARSE_EML_API_URL, files={"file": f})

        if response.status_code != 200:
            print(f"❌ Could not parse EML '{file_path}': {response.text}")
            return None

        data = response.json()
        email_obj = data.get("email")
        if not email_obj:
            print(f"❌ Invalid parse response for '{file_path}': {response.text}")
            return None

        return email_obj
    except Exception as e:
        print(f"❌ Exception parsing EML '{file_path}': {e}")
        return None

def insert_email(email_data: dict, email_type: str) -> bool:
    """Insert email into the vector database"""
    try:
        payload = {
            "subject": email_data.get("subject", ""),
            "body": email_data.get("body", ""),
            "sender": email_data.get("sender", "unknown@example.com"),
            "type": email_type
        }

        response = requests.post(INSERT_API_URL, json=payload)
        if response.status_code == 200:
            return True
        else:
            print(f"❌ Insert failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception inserting email: {e}")
        return False

def analyze_email(email_data: dict, expected_type: str, filename: str) -> dict:
    """Analyze email and return results for CSV"""
    try:
        payload = {
            "subject": email_data.get("subject", ""),
            "body": email_data.get("body", ""),
            "sender": email_data.get("sender", "unknown@example.com")
        }

        response = requests.post(ANALYZE_API_URL, json=payload)
        if response.status_code != 200:
            return {
                "filename": filename,
                "confidence": "ERROR",
                "BadSim": None,
                "GoodSim": None,
                "PhishingScore": None,
                "ScanResult": "ERROR",
                "ExpectedResult": expected_type,
                "MisclassificationType": "AnalyzeFailed"
            }

        data = response.json()
        phishing_score = data.get("phishing_score", 0)
        confidence_level = data.get("confidence_level", "Unknown")
        reasons = data.get("reasons", [])

        # Extract similarity scores from reasons
        bad_sim = None
        good_sim = None
        for reason in reasons:
            if "weighted_bad_score=" in reason:
                bad_sim = float(reason.split("=")[1])
            elif "weighted_good_score=" in reason:
                good_sim = float(reason.split("=")[1])

        # Determine prediction based on threshold
        predicted_type = "spam" if phishing_score >= 60 else "legitimate"

        # Determine correctness
        if predicted_type == expected_type:
            misclass_type = "Correct"
        elif predicted_type == "spam" and expected_type == "legitimate":
            misclass_type = "FalsePositive"
        else:
            misclass_type = "FalseNegative"

        return {
            "filename": filename,
            "confidence": confidence_level,
            "BadSim": bad_sim,
            "GoodSim": good_sim,
            "PhishingScore": phishing_score,
            "ScanResult": predicted_type,
            "ExpectedResult": expected_type,
            "MisclassificationType": misclass_type
        }
    except Exception as e:
        print(f"❌ Exception analyzing '{filename}': {e}")
        return {
            "filename": filename,
            "confidence": "ERROR",
            "BadSim": None,
            "GoodSim": None,
            "PhishingScore": None,
            "ScanResult": "ERROR",
            "ExpectedResult": expected_type,
            "MisclassificationType": "AnalyzeException"
        }

def create_sample_eml_files() -> int:
    """Create sample EML files if repositories don't have enough data"""
    samples_dir = Path(DATASET_DIR) / "downloads" / "manual_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Sample spam email
    spam_eml = """Return-Path: <noreply@suspicious-site.com>
Received: by mail.example.com with SMTP id abc123
Date: Mon, 15 Jan 2024 10:30:00 +0000
From: "Urgent Security Alert" <noreply@suspicious-site.com>
To: user@example.com
Subject: URGENT: Your Account Has Been Compromised!
Message-ID: <spam001@suspicious-site.com>
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8

URGENT SECURITY ALERT!

Your account has been compromised. Click here immediately to secure your account:
http://suspicious-site.com/fake-login

You have 24 hours to respond or your account will be permanently suspended.

This is not a drill. Act now!

Best regards,
Security Team (NOT REAL)
"""

    # Sample legitimate email
    legitimate_eml = """Return-Path: <newsletter@company.com>
Received: by mail.example.com with SMTP id def456
Date: Mon, 15 Jan 2024 14:15:00 +0000
From: "Company Newsletter" <newsletter@company.com>
To: user@example.com
Subject: Weekly Newsletter - Product Updates
Message-ID: <news001@company.com>
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8

Hello,

Here are this week's product updates:

1. New feature released in our mobile app
2. Upcoming maintenance scheduled for this weekend
3. Customer testimonials and success stories

Visit our website for more details: https://company.com

Best regards,
The Company Team
"""

    # Write sample files
    spam_file = samples_dir / "spam_sample.eml"
    legitimate_file = samples_dir / "legitimate_sample.eml"

    with open(spam_file, 'w', encoding='utf-8') as f:
        f.write(spam_eml)

    with open(legitimate_file, 'w', encoding='utf-8') as f:
        f.write(legitimate_eml)

    print(f"✅ Created 2 sample EML files in {samples_dir}")
    return 2

def download_dataset(dataset_key: str, dataset_info: dict) -> int:
    """Download EML files from a dataset"""
    print(f"\n📂 Processing dataset: {dataset_info['name']}")

    # Handle manual samples
    if dataset_info["type"] == "manual":
        return create_sample_eml_files()

    # Skip if no repository specified
    if not dataset_info.get("repo"):
        print(f"⚠️  No repository specified for {dataset_key}")
        return 0

    # Create dataset-specific directory
    dataset_dir = Path(DATASET_DIR) / "downloads" / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Fetch available EML files
    path = dataset_info.get("path", "")
    eml_files = fetch_github_files(dataset_info["repo"], path)

    if not eml_files:
        print(f"❌ No EML files found in {dataset_info['repo']}")
        return 0

    print(f"📋 Found {len(eml_files)} EML files")

    # Download files
    downloaded_count = 0
    for file_info in eml_files:
        local_path = dataset_dir / file_info['name']
        if download_file(file_info['download_url'], str(local_path)):
            downloaded_count += 1

    print(f"✅ Downloaded {downloaded_count} EML files to {dataset_dir}")
    return downloaded_count

def process_dataset_files(dataset_key: str, dataset_info: dict, do_insert: bool = True, do_analyze: bool = False) -> int:
    """Process EML files from a downloaded dataset"""
    dataset_dir = Path(DATASET_DIR) / "downloads" / dataset_key

    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return 0

    # Get all EML files
    eml_files = list(dataset_dir.glob("*.eml"))
    if not eml_files:
        print(f"❌ No EML files found in {dataset_dir}")
        return 0

    print(f"📁 Processing {len(eml_files)} EML files from {dataset_info['name']}")

    processed_count = 0
    analyze_tasks = []

    # Process each file
    for eml_file in eml_files:
        email_data = parse_eml_file(str(eml_file))
        if not email_data:
            continue

        # Determine email type based on dataset
        if dataset_info["type"] == "spam":
            email_type = "spam"
            expected_type = "spam"
        elif dataset_info["type"] == "legitimate":
            email_type = "business"
            expected_type = "legitimate"
        else:  # mixed - try to infer from filename
            if "spam" in eml_file.name.lower() or "phish" in eml_file.name.lower():
                email_type = "spam"
                expected_type = "spam"
            else:
                email_type = "business"
                expected_type = "legitimate"

        # Insert training data
        if do_insert:
            if insert_email(email_data, email_type):
                processed_count += 1

        # Prepare for analysis
        if do_analyze:
            analyze_tasks.append((email_data, expected_type, eml_file.name))

    # Analyze emails in parallel
    if do_analyze and analyze_tasks:
        print(f"🔍 Analyzing {len(analyze_tasks)} emails...")
        with ThreadPoolExecutor(max_workers=4) as executor, tqdm(total=len(analyze_tasks), desc="Analyzing", unit="emails") as pbar:
            futures = [executor.submit(analyze_email, email_data, expected_type, filename)
                      for email_data, expected_type, filename in analyze_tasks]

            for future in as_completed(futures):
                result = future.result()
                if result:
                    analysis_results.append(result)
                pbar.update(1)

    return processed_count

def write_results_to_csv(results: list, output_file: str = "eml_test_results.csv"):
    """Write analysis results to CSV"""
    if not results:
        print("❌ No results to write")
        return

    fieldnames = [
        "filename",
        "confidence",
        "BadSim",
        "GoodSim",
        "PhishingScore",
        "ScanResult",
        "ExpectedResult",
        "MisclassificationType"
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Analysis results written to {output_file}")

def calculate_metrics(results: list):
    """Calculate and display performance metrics"""
    if not results:
        print("❌ No results to analyze")
        return

    # Filter out error cases
    valid_results = [r for r in results if r["MisclassificationType"] not in ("AnalyzeFailed", "AnalyzeException", "ERROR")]

    if not valid_results:
        print("❌ No valid test results")
        return

    total_tested = len(valid_results)
    correct = sum(1 for r in valid_results if r["MisclassificationType"] == "Correct")
    false_positives = sum(1 for r in valid_results if r["MisclassificationType"] == "FalsePositive")
    false_negatives = sum(1 for r in valid_results if r["MisclassificationType"] == "FalseNegative")

    # Confidence stats
    high_conf = sum(1 for r in valid_results if r["confidence"] == "High")
    medium_conf = sum(1 for r in valid_results if r["confidence"] == "Medium")
    low_conf = sum(1 for r in valid_results if r["confidence"] == "Low")

    # Similarity stats
    bad_sims = [r["BadSim"] for r in valid_results if r["BadSim"] is not None]
    good_sims = [r["GoodSim"] for r in valid_results if r["GoodSim"] is not None]

    # Calculate percentages
    accuracy = (correct / total_tested) * 100
    fp_rate = (false_positives / total_tested) * 100
    fn_rate = (false_negatives / total_tested) * 100
    high_conf_rate = (high_conf / total_tested) * 100
    medium_conf_rate = (medium_conf / total_tested) * 100
    low_conf_rate = (low_conf / total_tested) * 100

    print(f"\n📊 EML Dataset Test Summary:")
    print(f"Total Emails Tested: {total_tested}")
    print(f"Correct Classifications: {correct} ({accuracy:.2f}%)")
    print(f"False Positives: {false_positives} ({fp_rate:.2f}%)")
    print(f"False Negatives: {false_negatives} ({fn_rate:.2f}%)")

    print(f"\n🔍 Confidence Level Breakdown:")
    print(f"High Confidence: {high_conf} ({high_conf_rate:.2f}%)")
    print(f"Medium Confidence: {medium_conf} ({medium_conf_rate:.2f}%)")
    print(f"Low Confidence: {low_conf} ({low_conf_rate:.2f}%)")

    if bad_sims and good_sims:
        avg_bad = sum(bad_sims) / len(bad_sims)
        avg_good = sum(good_sims) / len(good_sims)
        print(f"\n🔎 Similarity Statistics:")
        print(f"Avg BadSim: {avg_bad:.3f}")
        print(f"Avg GoodSim: {avg_good:.3f}")

def main():
    """Main function"""
    print("🚀 EML Dataset Tester for VectorShield")
    print("=" * 50)
    print(f"Test session: {DOWNLOAD_DATE}")
    print()

    # Create directories
    create_directories()

    # Step 1: Download datasets
    print("=== STEP 1: DOWNLOADING EML DATASETS ===")
    total_downloaded = 0

    for dataset_key, dataset_info in DATASETS.items():
        downloaded = download_dataset(dataset_key, dataset_info)
        total_downloaded += downloaded

    if total_downloaded == 0:
        print("❌ No EML files were downloaded. Exiting.")
        return

    print(f"\n✅ Downloaded {total_downloaded} EML files total")

    # Step 2: Import training data
    print("\n=== STEP 2: IMPORTING TRAINING DATA ===")
    total_imported = 0

    for dataset_key, dataset_info in DATASETS.items():
        imported = process_dataset_files(dataset_key, dataset_info, do_insert=True, do_analyze=False)
        total_imported += imported
        print(f"✅ Imported {imported} emails from {dataset_info['name']}")

    if total_imported == 0:
        print("❌ No emails were imported. Check API connectivity.")
        return

    print(f"\n✅ Total emails imported: {total_imported}")

    # Wait for batch processing
    print("⏳ Waiting 10 seconds for batch upserts to complete...")
    time.sleep(10)

    # Step 3: Test/analyze data
    print("\n=== STEP 3: TESTING MODEL PREDICTIONS ===")

    for dataset_key, dataset_info in DATASETS.items():
        process_dataset_files(dataset_key, dataset_info, do_insert=False, do_analyze=True)

    if not analysis_results:
        print("❌ No emails were analyzed. Check dataset files.")
        return

    # Step 4: Generate reports
    print("\n=== STEP 4: GENERATING REPORTS ===")

    # Write CSV
    output_file = f"eml_test_results_{DOWNLOAD_DATE}.csv"
    write_results_to_csv(analysis_results, output_file)

    # Calculate metrics
    calculate_metrics(analysis_results)

    print(f"\n✅ EML Dataset Testing Complete!")
    print(f"📄 Results saved to: {output_file}")

if __name__ == "__main__":
    main()