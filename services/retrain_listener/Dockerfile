FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY retrain_listener.py .

CMD ["python", "retrain_listener.py"] 