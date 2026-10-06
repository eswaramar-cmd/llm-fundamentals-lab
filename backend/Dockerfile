FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch
RUN pip install --no-cache-dir torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy agent
COPY agent/ ./agent/

# Copy FastAPI application
COPY backend/main.py ./main.py

# Persistent directories
RUN mkdir -p /chroma_rag /chroma_memory /documents

# Ensure Python output is unbuffered for Docker logs
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]