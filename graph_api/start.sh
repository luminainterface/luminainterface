#!/bin/bash

# Set Python path
export PYTHONPATH=/app

# Start the service using Python directly
exec python -m uvicorn main:app --host 0.0.0.0 --port 8000 