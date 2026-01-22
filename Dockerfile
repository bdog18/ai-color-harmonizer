FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps: add OpenCV runtime libs (safe even if you use headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libice6 \
    libfontconfig1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit config for container environments
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

CMD streamlit run app/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port ${PORT:-8080}
