# 🔹 Stage 1: Download Model
FROM python:3.12-slim AS model-downloader

WORKDIR /model

# Install only the necessary package for downloading the model
RUN pip install --no-cache-dir sentence-transformers

# Download and save the model
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('all-MiniLM-L6-v2'); \
    model.save('/model/all-MiniLM-L6-v2')"

# 🔹 Stage 2: Build Final Application Image
FROM python:3.12-slim

WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .

# Install dependencies (without cache for smaller image)
RUN pip install --no-cache-dir -r requirements.txt

# 🔹 Copy the downloaded model from the previous stage
COPY --from=model-downloader /model/all-MiniLM-L6-v2 models/all-MiniLM-L6-v2

# Copy the rest of the application
COPY . .

# Create a non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
USER appuser

# Expose port for API
EXPOSE 5000

# Use Gunicorn for better performance (adjust workers as needed)
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
# uvicorn app.main:app --host 0.0.0.0 --port 5000
