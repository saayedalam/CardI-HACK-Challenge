# ---- base ----
FROM python:3.12-slim

# System deps (lightgbm needs gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy only lockfile first for better caching
COPY docs/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the project
COPY . /app

# Default command builds the submission (expects data in /app/data/raw)
ENV PYTHONPATH=/app
CMD ["python", "make_submission.py"]
