# 🧪 V7 Logic Agent Testing Suite
# Comprehensive Validation for Classification, Einstein Puzzles, and Logic Capabilities

FROM python:3.11-slim

LABEL maintainer="Ultimate AI Orchestration Architecture v10"
LABEL service="v7-logic-agent-tester"
LABEL version="v7.1.0"
LABEL purpose="comprehensive-validation"
LABEL test_categories="classification,einstein-puzzles,logic-puzzles,mathematical,ai-failsafe"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-v7.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy test scripts
COPY test_classification_fix.py /app/
COPY test_einstein_puzzle.py /app/
COPY test_logic_puzzles_v7.py /app/

# Create comprehensive test runner
COPY v7_comprehensive_tester.py /app/

# Create directories for test results and reports
RUN mkdir -p /app/test_results /app/reports /app/logs

# Environment variables for testing configuration
ENV ENABLE_CLASSIFICATION_TESTS=true
ENV ENABLE_EINSTEIN_PUZZLE_TESTS=true
ENV ENABLE_LOGIC_PUZZLE_TESTS=true
ENV ENABLE_MATHEMATICAL_TESTS=true
ENV ENABLE_AI_FAILSAFE_TESTS=true
ENV TARGET_CLASSIFICATION_ACCURACY=91.7
ENV TARGET_EINSTEIN_SUCCESS_RATE=100.0
ENV TARGET_MATHEMATICAL_PRECISION=100.0

# Default test run command
CMD ["python", "v7_comprehensive_tester.py"] 