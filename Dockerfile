FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    build-essential \
    python3-dev \
    ffmpeg \
    wget \
    execstack \
    && rm -rf /var/lib/apt/lists/*

# python 심볼릭 링크
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install pip==23.2.1 setuptools==68.2.2 wheel==0.41.2

# PyTorch GPU 버전 설치 (CUDA 11.8)
RUN pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 나머지 requirements 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8080

# Run the application
CMD ["python", "video_processing_tasks.py"]