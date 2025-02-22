import os
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------
# 📁 Setup File Paths
# -------------------------------
current_dir = os.path.dirname(os.path.abspath("__file__"))  # Use os.getcwd() if running in Jupyter
csv_file_path = os.path.join(current_dir, "dataset", "enron_data_fraud_labeled.csv")

# -------------------------------
# 📦 Load Dataset
# -------------------------------
data = pd.read_csv(csv_file_path)
print(f"Dataset loaded successfully! {len(data)} rows found.")

# -------------------------------
# 📌 Ensure Data Integrity
# -------------------------------
# Check if required columns exist
required_columns = {"Body", "Subject", "From", "Label"}
if not required_columns.issubset(data.columns):
    raise ValueError(f"Dataset is missing required columns: {required_columns - set(data.columns)}")

# -------------------------------
# 🔀 Shuffle the Data
# -------------------------------
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle dataset

# -------------------------------
# 📊 Preserve Spam-to-Ham Ratio
# -------------------------------
train_data, test_data = train_test_split(
    data,
    test_size=0.2,  # 20% for testing
    stratify=data["Label"],  # Maintain spam/ham ratio
    random_state=42
)

# -------------------------------
# 💾 Save Split Data
# -------------------------------
train_file_path = os.path.join(current_dir, "dataset", "train_data.csv")
test_file_path = os.path.join(current_dir, "dataset", "test_data.csv")

train_data.to_csv(train_file_path, index=False)
test_data.to_csv(test_file_path, index=False)

print(f"✅ Data split completed! Training set: {len(train_data)} rows, Test set: {len(test_data)} rows")
print(f"📂 Train data saved to: {train_file_path}")
print(f"📂 Test data saved to: {test_file_path}")
