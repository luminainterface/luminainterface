FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py ranker.py ./
EXPOSE 8601
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8601"] 