# VectorShield: Advanced Spam Detection with a Vector Database
![VectorShield Logo](./logo.png)

VectorShield is a modern spam detection engine that enhances traditional scanners by leveraging **vector similarity search**. It uses machine learning to detect **phishing, spam, and suspicious emails** that share structural or semantic similarities with previously flagged content.

Unlike conventional spam filters, VectorShield doesn't rely on static rules or keyword lists. Instead, it compares the **vector embeddings** of incoming emails against a **Qdrant vector database** to find close matches, making it more resilient against obfuscation and trickery.

---

## Features
- **⚡ Real-Time Vector Analysis**: Transforms emails into embeddings and compares them against known phishing/spam vectors.
- **🧠 Continual Learning**: Improves detection accuracy over time by learning from legitimate and malicious feedback.
- **📈 Customizable Threshold**: Define your own `BAD_PROB_THRESHOLD` to adjust sensitivity.
- **🧪 Accurate Similarity Scoring**: Produces a `phishing_score` (0–100) with labeled reasoning.
- **📬 Easy Integration**: RESTful API for inserting, analyzing, and managing email classification.
- **🛠️ False Positive Removal**: Use the `/report_false_positive` API to clean up incorrectly flagged emails.
- **📊 Prometheus Metrics**: Observe system performance and request metrics out-of-the-box.

---

## Dataset Tools
Inside the [`dataset/`](./dataset) folder you'll find utilities to evaluate the system using the **Enron Fraud Email Dataset**:

### 1. Download Dataset
```bash
python dataset/download-enron_data_fraud_labeled.py
```
You will be prompted to download the CSV manually from Kaggle and place it in the correct folder.

### 2. Import Emails into Vector DB
```bash
python dataset/import-enron_data_fraud_labeled.py
```
- Converts CSV into email vectors
- Sends data to the `/insert` API
- Supports resume from last progress

### 3. Evaluate Model Performance
```bash
python dataset/test-enron_data_fraud_labeled.py
```
Outputs include:
```
📊 Test Summary:
Total Emails Tested: 5000
Correct Classifications: 4782
False Positives: 105
False Negatives: 113
Accuracy: 95.64%
```

---

## API Endpoints

### 1. `/insert`
Insert a labeled email into the system.
```json
POST /insert
{
  "subject": "Security Alert",
  "body": "VGhpcyBpcyBhIHRlc3QgZW1haWwu",
  "sender": "alerts@example.com",
  "type": "spam"
}
```
Response:
```json
{
  "message": "✅ Queued BAD email [spam]: Security Alert"
}
```

### 2. `/analyze`
Analyze an email and get its phishing score.
```json
POST /analyze
{
  "subject": "Win a Free Cruise!",
  "body": "VGhpcyBpcyBhIGZha2UgZW1haWwgYm9keS4=",
  "sender": "cruise@scamsite.com"
}
```
Response:
```json
{
  "phishing_score": 85,
  "confidence_level": "High",
  "closest_match": "bad",
  "reasons": [
    "sum_good_sim=3.22",
    "sum_bad_sim=11.44",
    "Top match => bad, sub_label=spam"
  ]
}
```

### 3. `/report_false_positive`
Remove an incorrectly flagged email.
```json
POST /report_false_positive
{
  "subject": "Monthly Newsletter",
  "body": "VGhpcyBpcyBub3Qgc3BhbS4=",
  "sender": "news@example.com"
}
```

### 4. `/parse_eml`
Upload `.eml` files directly and extract subject, sender, and base64 body.
```bash
curl -F "file=@sample.eml" http://localhost:5000/parse_eml
```
Response:
```json
{
  "message": "Parsed EML",
  "email": {
    "subject": "Test Email",
    "sender": "sender@example.com",
    "body": "...base64..."
  }
}
```

---

## How It Works
1. Emails are vectorized with `sentence-transformers`.
2. Stored in Qdrant with label = good or bad.
3. New emails are compared via similarity search.
4. A score is computed: `bad_score = sum_bad / (sum_good + sum_bad)`
5. The `BAD_PROB_THRESHOLD` determines classification.

---

## Getting Started

### Requirements
- Python 3.9+
- Qdrant (via Docker or external)
- FastAPI, Uvicorn, PyTorch, Transformers

### Setup
```bash
git clone https://github.com/aeggerd/VectorShield.git
cd VectorShield
pip install -r requirements.txt
docker-compose up -d  # start Qdrant
uvicorn app.main:app --reload --host 0.0.0.0 --port 5000
```
Then open: http://localhost:5000/docs

---

## Favicon / Web UI
To add a favicon for your browser tab:
1. Copy the file `static/favicon.ico` to your project
2. Add this to your `<head>` in the HTML:
```html
<link rel="icon" href="/static/favicon.ico" type="image/x-icon">
```

---

## License
Apache License 2.0. See the [LICENSE](LICENSE) file.

---

## Roadmap
- Support additional vector backends: Pinecone, Milvus
- UI: Visualize spam clusters and matches
- Auto-retraining based on user feedback
- Confidence-based alerting thresholds
