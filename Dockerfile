FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including file locking support)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    util-linux \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the PDF file into the Docker image
COPY random_machine_learing.pdf /app/random_machine_learing.pdf

# Copy application code
COPY Chatbot.py .
COPY Langgraph_Agent.py .
COPY helper_functions.py .
COPY .env* ./

# Create directories for data persistence with proper permissions
RUN mkdir -p /app/emissions_data /app/training_data \
    && chmod 755 /app/emissions_data /app/training_data

# Create the training data file with proper permissions
RUN touch /app/training_data.jsonl \
    && chmod 664 /app/training_data.jsonl

# Create non-root user for security (optional but recommended)
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose Streamlit and Prometheus ports
EXPOSE 8501 8080

# Enhanced health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Accept build argument
ARG ARCHITECTURE_VERSION=3

# Set environment variable from build argument
ENV ARCHITECTURE_VERSION=${ARCHITECTURE_VERSION}

# Set default pod identification (will be overridden by Kubernetes)
ENV POD_NAME=local
ENV POD_IP=localhost

# Run Streamlit app with version parameter
CMD ["sh", "-c", "streamlit run Chatbot.py --server.port=8501 --server.address=0.0.0.0 -- -v ${ARCHITECTURE_VERSION}"]