FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY model ./model

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
  CMD curl --fail http://localhost:8000/ || exit 1

CMD ["sh", "-c", "python download_model.py && exec uvicorn app:app --host 0.0.0.0 --port 8000"]

# CMD ["sh", "-c", "python download_model.py && uvicorn app:app --host 0.0.0.0 --port 8000"]