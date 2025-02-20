FROM python:3.12-slim

WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .

# Install dependencies (without cache for smaller image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
USER appuser

# Expose port for API
EXPOSE 8000

# Use Gunicorn for better performance (adjust workers as needed)
ENTRYPOINT ["uvicorn", "web_ui:app", "--host", "0.0.0.0", "--port", "8000"]
