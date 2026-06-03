# Python 3.11 to match CI and the version the lockfile was resolved against.
FROM python:3.14-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    make \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install pinned dependencies first (better caching, deterministic build)
COPY requirements.lock .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.lock

# Copy source code
COPY . .

RUN useradd -m appuser
USER appuser

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
