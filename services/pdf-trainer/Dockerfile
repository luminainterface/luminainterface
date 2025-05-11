FROM python:3.11-slim
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/training_data

ENV PYTHONUNBUFFERED=1
EXPOSE 8826

# Run both the FastAPI health server and the continuous PDF trainer
CMD ["sh", "-c", "python /app/pdf_trainer.py 2>&1 | tee /app/pdf_trainer.log"] 