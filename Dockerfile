cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Cache locations inside the IMAGE
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV SENTENCE_TRANSFORMERS_HOME=/models

# Make sure cache folder exists
RUN mkdir -p /models

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ PRELOAD MODEL INTO /models DURING BUILD (CLOUD-BUILD SAFE)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/models'); print('Model cached successfully')"

# Copy app last
COPY . .

EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
EOF
