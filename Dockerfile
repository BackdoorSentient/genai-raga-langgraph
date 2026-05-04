FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY main.py main.py

RUN mkdir -p data/raw data/processed data/vector_store

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]