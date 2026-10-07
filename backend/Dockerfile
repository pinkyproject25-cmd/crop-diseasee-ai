FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY models ./models
COPY install_candidate_v2.py fetch_model_artifacts.py ./
CMD ["sh", "-c", "python fetch_model_artifacts.py && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]