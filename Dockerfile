FROM python:3.11-slim

WORKDIR /app

# ---- ENV: force offline + cache location ----
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV SENTENCE_TRANSFORMERS_HOME=/models
ENV TRANSFORMERS_OFFLINE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ---- PRELOAD MODEL INTO CACHE ----
RUN python - <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="/models"
)
print("Model cached successfully")
EOF

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
