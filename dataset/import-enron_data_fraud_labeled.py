import os
import pandas as pd
import base64
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

# We'll use a lock to avoid race conditions when multiple threads write progress
progress_lock = threading.Lock()
global_progress = get_last_processed_index()

def update_progress_if_higher(index):
    global global_progress
    with progress_lock:
        if index > global_progress:
            global_progress = index
            with open(progress_file_path, "w") as file:
                file.write(str(index))

# -------------------------------
# 🏭 Worker Function
# -------------------------------
def process_row(df_index, row, insert_api_url):
    """
    Submits a single row to the /insert endpoint.
    Returns (df_index, success_bool).
    """
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

        resp = requests.post(insert_api_url, json=payload)
        if resp.status_code == 200:
            return (df_index, True)
        else:
            print(f"❌ Error inserting row {df_index}: {resp.status_code} - {resp.text}")
            return (df_index, False)

    except Exception as e:
        print(f"❌ Exception at row {df_index}: {e}")
        return (df_index, False)

# -------------------------------
# 🚀 Process Rows in Parallel
# -------------------------------
insert_api_url = "http://localhost:5000/insert"
start_index = global_progress
print(f"Resuming from row {start_index}...")

total_rows = len(data)
rows_to_process = data.iloc[start_index:].iterrows()

with ThreadPoolExecutor(max_workers=4) as executor, \
     tqdm(total=(total_rows - start_index), desc="Inserting", unit="rows") as pbar:

    futures_map = {}
    for df_index, row in rows_to_process:
        future = executor.submit(process_row, df_index, row, insert_api_url)
        futures_map[future] = df_index

    for future in as_completed(futures_map):
        df_index, success = future.result()
        if success:
            update_progress_if_higher(df_index + 1)
        pbar.update(1)
        pbar.refresh()

print("\n✅ Import complete.")
