import os
import pandas as pd
import zipfile
import requests
import base64

# -------------------------------
# 📁 Setup File Paths
# -------------------------------
current_dir = os.path.dirname(os.path.abspath("__file__"))  # Use os.getcwd() if running in Jupyter
# zip_file_path = os.path.join(current_dir, "dataset", "enron_data_fraud_labeled.csv.zip")
csv_file_path = os.path.join(current_dir, "dataset",  "enron_data_fraud_labeled.csv")
progress_file_path = os.path.join(current_dir, "dataset", "progress.txt")  # File to track the last processed row

# -------------------------------
# 📦 Extract and Load Dataset
# -------------------------------
# if not os.path.exists(csv_file_path):
#     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#         zip_ref.extractall(current_dir)

data = pd.read_csv(csv_file_path)
print("Dataset loaded successfully!")

# -------------------------------
# 📌 Helper Functions
# -------------------------------
def get_email_type(label):
    """Determines email type based on the 'Label' field."""
    return "phishing" if label == 1 else "legitimate"

def get_last_processed_index():
    """Reads the last processed row index from the progress file."""
    if os.path.exists(progress_file_path):
        with open(progress_file_path, "r") as file:
            return int(file.read().strip())
    return 0  # Start from the beginning if no progress file exists

def save_progress(index):
    """Saves the current row index to the progress file."""
    with open(progress_file_path, "w") as file:
        file.write(str(index))

# -------------------------------
# 🚀 Process Rows and Send Requests
# -------------------------------
insert_api_url = "http://localhost:5000/insert"
start_index = get_last_processed_index()
print(f"Resuming from row {start_index}...")

for index, row in data.iloc[start_index:].iterrows():
    try:
        email_body = row["Body"]
        email_subject = row["Subject"] if pd.notna(row["Subject"]) else "No Subject"
        email_sender = row["From"] if pd.notna(row["From"]) else "unknown@enron.com"
        email_type = get_email_type(row["Label"])

        # Prepare the request payload
        payload = {
            "subject": email_subject,
            "body": base64.b64encode(email_body.encode("utf-8")).decode("utf-8"),
            "sender": email_sender,
            "type": email_type
        }

        # Send the POST request
        response = requests.post(insert_api_url, json=payload)

        # Handle the response
        if response.status_code == 200:
            print(f"Row {index} inserted successfully: {response.json()['message']}")
        else:
            print(f"Error inserting row {index}: {response.status_code} - {response.text}")

        # Save progress after each successful request
        save_progress(index + 1)

    except Exception as e:
        print(f"An error occurred at row {index}: {e}")
        break  # Stop the loop on error to avoid losing track of progress