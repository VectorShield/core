import os
import pandas as pd
import zipfile
import requests
import base64
from tqdm import tqdm  # <-- ADD THIS for the progress bar

# -------------------------------
# 📁 Setup File Paths
# -------------------------------
current_dir = os.path.dirname(os.path.abspath("__file__"))
csv_file_path = os.path.join(current_dir, "dataset", "train_data-enron_data_fraud_labeled.csv")
progress_file_path = os.path.join(current_dir, "dataset", "progress.txt")

# -------------------------------
# 📦 Load Dataset
# -------------------------------
data = pd.read_csv(csv_file_path)
print("Dataset loaded successfully!")

# -------------------------------
# 📌 Helper Functions
# -------------------------------
def get_email_type(label):
    return "phishing" if label == 1 else "legitimate"

def get_last_processed_index():
    if os.path.exists(progress_file_path):
        with open(progress_file_path, "r") as file:
            return int(file.read().strip())
    return 0

def save_progress(index):
    with open(progress_file_path, "w") as file:
        file.write(str(index))

# -------------------------------
# 🚀 Process Rows and Send Requests
# -------------------------------
insert_api_url = "http://localhost:5000/insert"
start_index = get_last_processed_index()
print(f"Resuming from row {start_index}...")

total_rows = len(data)

# Use tqdm to show a progress bar and ETA
with tqdm(total=(total_rows - start_index), desc="Inserting", unit="rows") as pbar:
    # We slice the DataFrame from 'start_index' onward
    for df_index, row in data.iloc[start_index:].iterrows():
        try:
            email_body = row["Body"]
            email_subject = row["Subject"] if pd.notna(row["Subject"]) else "No Subject"
            email_sender = row["From"] if pd.notna(row["From"]) else "unknown@enron.com"
            email_type = get_email_type(row["Label"])

            payload = {
                "subject": email_subject,
                "body": base64.b64encode(email_body.encode("utf-8")).decode("utf-8"),
                "sender": email_sender,
                "type": email_type
            }

            response = requests.post(insert_api_url, json=payload)

            if response.status_code != 200:
                # print(f"Row {df_index} inserted: {response.json().get('message','')}")
            # else:
                print(f"Error inserting row {df_index}: {response.status_code} - {response.text}")

            # Save progress and update the bar
            save_progress(df_index + 1)
            pbar.update(1)

        except Exception as e:
            print(f"An error occurred at row {df_index}: {e}")
            break
