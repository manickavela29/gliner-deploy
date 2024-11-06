# Use NVIDIA's Triton Inference Server base image with the specified version
FROM nvcr.io/nvidia/tritonserver:24.10-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL_REPO_PATH=/models

# Copy the model repository into the container
COPY python_model_repository ${MODEL_REPO_PATH}

# Install necessary Python dependencies
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --no-cache-dir torch torchvision transformers gliner && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Expose Triton server ports
EXPOSE 8000 8001 8002

# Command to start Triton Inference Server
CMD ["tritonserver", "--model-repository=/models"]