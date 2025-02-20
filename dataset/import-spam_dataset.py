#%% import first

# Import necessary libraries
import os
import pandas as pd
import requests
import base64

#%% load spam_dataset.csv

# Load the dataset
file_path = f"{os.path.dirname(os.path.abspath(__file__))}/spam_dataset.csv"  # Adjust the path if needed
data = pd.read_csv(file_path)

# Display the first few rows to confirm the structure
print("Dataset preview:")
print(data.head())

# API endpoint
insert_api_url = "http://localhost:5000/insert"

# Function to determine email type based on 'is_spam' value
def get_email_type(is_spam):
    return "phishing" if is_spam == 1 else "legitimate"

# Iterate over each row and send a POST request to the API
for index, row in data.iterrows():
    email_content = row["message_content"]
    email_type = get_email_type(row["is_spam"])
    
    # Prepare the request payload
    payload = {
        "subject": "",
        "body": base64.b64encode(email_content.encode("utf-8")).decode("utf-8"),
        "sender": "",
        "customerId": "99999",
        "type": email_type
    }

    # Send the POST request
    response = requests.post(insert_api_url, json=payload)

    # Print response status
    if response.status_code == 200:
        print(f"Row {index} inserted successfully: {response.json()['message']}")
    else:
        print(f"Error inserting row {index}: {response.status_code} - {response.text}")

