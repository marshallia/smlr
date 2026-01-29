# Use Ubuntu base image
FROM ubuntu:22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git wget curl \
    build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y locales \
    && locale-gen zh_TW.UTF-8 \
    && update-locale LANG=zh_TW.UTF-8 \

# Set Python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create working directory
WORKDIR /app

COPY requirements.txt /app

RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# Copy your project files into the container
COPY . /app