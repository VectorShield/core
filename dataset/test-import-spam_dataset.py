#%% Import libraries
import os
import pandas as pd
import requests
import base64

#%% Load dataset
file_path = f"{os.path.dirname(os.path.abspath(__file__))}/spam_dataset.csv"
data = pd.read_csv(file_path)

# Display the first few rows to confirm the structure
print("Dataset preview:")
print(data.head())

# API endpoint for analyzing emails
analyze_api_url = "http://localhost:5000/analyze"

# Function to determine email type based on 'is_spam' value
def get_expected_label(is_spam):
    return "phishing" if is_spam == 1 else "legitimate"

# Variables to track results
total_emails = 0
correctly_classified = 0
false_positives = 0
false_negatives = 0

results = []

#%% Iterate over each row and send a POST request to the API
for index, row in data.iterrows():
    total_emails += 1
    email_content = row["message_content"]
    expected_label = get_expected_label(row["is_spam"])
    
    # Prepare the request payload for analysis
    payload = {
        "subject": "",
        "body": base64.b64encode(email_content.encode("utf-8")).decode("utf-8"),
        "sender": ""
    }

    # Send the POST request to analyze the email
    response = requests.post(analyze_api_url, json=payload)

    if response.status_code == 200:
        result = response.json()
        predicted_label = "phishing" if result["phishing_score"] >= 70 else "legitimate"
        
        # Check if the prediction matches the expected label
        is_correct = (predicted_label == expected_label)
        correctly_classified += 1 if is_correct else 0
        
        # Track false positives and false negatives
        if predicted_label == "phishing" and expected_label == "legitimate":
            false_positives += 1
        elif predicted_label == "legitimate" and expected_label == "phishing":
            false_negatives += 1
        
        # Append result for later review
        results.append({
            "index": index,
            "expected_label": expected_label,
            "predicted_label": predicted_label,
            "confidence_level": result["confidence_level"],
            "phishing_score": result["phishing_score"],
            "is_correct": is_correct
        })
        
        print(f"Row {index}: Expected: {expected_label}, Predicted: {predicted_label}, Score: {result['phishing_score']}%")
    else:
        print(f"Error analyzing row {index}: {response.status_code} - {response.text}")

#%% Calculate and display statistics
accuracy = (correctly_classified / total_emails) * 100
false_positive_rate = (false_positives / total_emails) * 100
false_negative_rate = (false_negatives / total_emails) * 100

print("\n===================================")
print(f"Total Emails: {total_emails}")
print(f"Correctly Classified: {correctly_classified}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"False Positives: {false_positives} ({false_positive_rate:.2f}%)")
print(f"False Negatives: {false_negatives} ({false_negative_rate:.2f}%)")
print("===================================\n")

#%% Save results to a CSV for further review
results_df = pd.DataFrame(results)
results_df.to_csv(f"{os.path.dirname(file_path)}/validation_results.csv", index=False)
print("Results saved to validation_results.csv.")
