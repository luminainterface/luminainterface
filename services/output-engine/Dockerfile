FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies and Python 3.11
RUN apt-get update && \
    apt-get install -y software-properties-common wget git build-essential lsb-release gnupg2 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-distutils python3-pip && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install OpenVINO dependencies
RUN wget https://apt.repos.intel.com/intel-gpg-keys/Intel-GPG-KEY-INTEL-SW-PRODUCTS.PUB && \
    apt-key add Intel-GPG-KEY-INTEL-SW-PRODUCTS.PUB && \
    echo "deb https://apt.repos.intel.com/openvino/2023 ubuntu22 main" > /etc/apt/sources.list.d/intel-openvino-2023.list && \
    apt-get update && \
    apt-get install -y intel-openvino-runtime-ubuntu22-2023.3.0 && \
    rm -rf /var/lib/apt/lists/*

# Set OpenVINO environment variables
ENV INTEL_OPENVINO_DIR=/opt/intel/openvino_2023
ENV LD_LIBRARY_PATH=/opt/intel/openvino_2023/runtime/lib/intel64:$LD_LIBRARY_PATH
ENV PATH=/opt/intel/openvino_2023/runtime/bin:$PATH

# Create models directory
RUN mkdir -p /app/models

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:${PYTHONPATH}" \
    TRANSFORMERS_CACHE="/app/models" \
    HF_HOME="/app/models" \
    OPENVINO_DEVICE="CPU" \
    USE_NPU="0"

# The model will be downloaded and converted to OpenVINO format on first run
# This is intentional to keep the image size smaller
# and allow for model updates without rebuilding

# Run the service
EXPOSE 8813
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8813"] 