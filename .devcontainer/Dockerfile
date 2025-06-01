FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt