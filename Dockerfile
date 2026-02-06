FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    ffmpeg \
    build-essential \
    execstack \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# pip 최신 버전
RUN pip install --upgrade pip

# PyTorch GPU 버전 설치 (CUDA 11.8)
RUN pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 나머지 requirements 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the application
CMD ["python", "video_processing_tasks.py"]