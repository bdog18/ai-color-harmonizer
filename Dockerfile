FROM python:3.11-slim

# Avoid Python writing .pyc files; unbuffer logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (lightweight, common)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Streamlit config for container environments
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

# Railway provides $PORT
CMD ["bash", "-lc", "streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT:-8080}"]
