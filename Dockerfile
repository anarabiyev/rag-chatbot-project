FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only PyTorch first: the default Linux wheel pulls in GPU libraries that add several GB
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# download the models now, so the running container needs no internet access
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" \
 && python -c "from transformers import pipeline; pipeline('text-classification', model='ProsusAI/finbert')"

# build the vector index from the PDFs (a few minutes on CPU). Only the files ingestion needs are copied
# first, so changing the agent code does not trigger a rebuild of the index.
COPY rag ./rag
COPY ingest.py .
COPY corpus ./corpus
RUN python ingest.py

COPY . .

# the LLM runs on the host (LM Studio); set LLM_BACKEND=dummy to run without it
ENV HF_HUB_OFFLINE=1 \
    LLM_BASE_URL=http://host.docker.internal:1234/v1

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
