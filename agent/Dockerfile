FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY agent/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agent/app/ /app/app/
COPY agent/.env.example /app/.env

# Expose port for API server
EXPOSE 8000

# Set environment variables
ENV DEBUG=0
ENV RAG_SERVICE_URL=http://rag:8001

# Run API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
