FROM python:3.13-slim

WORKDIR /app

# install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project code
COPY . .

# create data directories
RUN mkdir -p data/raw data/embeddings

# expose both ports (Streamlit + FastAPI)
EXPOSE 8501 8000

# default: run Streamlit
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
