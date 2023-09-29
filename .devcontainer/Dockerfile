
# Import environment
FROM mcr.microsoft.com/devcontainers/python:3.11

# Install TensorFlow + GPU support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers (huggingface) and other dependencies
RUN pip install transformers accelerate sentencepiece protobuf

# Set environment vars
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub