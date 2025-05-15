FROM python:3.11-slim

WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY .env .

# Create data directory
RUN mkdir -p /app/data

# Expose port
EXPOSE 6000

# Run application
CMD ["python", "app.py"]
