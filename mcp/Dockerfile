FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy MCP package and server code
COPY mcp/ /app/mcp/
COPY setup.py .

# Install the package
RUN pip install -e .

# Expose port for MCP SSE server
EXPOSE 8080

# Set environment variables
ENV DEBUG=0
ENV AGENT_SERVICE_URL=http://agent:8000

# Run MCP server
CMD ["python", "-m", "mcp.server"]
