# On-Premise Training Image
# Requires: NVIDIA GPU with CUDA 12.4+, 24GB+ VRAM (standard tier), 80GB+ (performance)

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN pip install --no-cache-dir \
    unsloth \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    transformers accelerate bitsandbytes \
    optimum onnxruntime-gpu \
    numpy pandas s3fs \
    peft trl

WORKDIR /app

COPY onprem/ /app/onprem/
COPY requirements.txt /app/

RUN mkdir -p /data /exports /checkpoints

ENTRYPOINT ["python", "/app/onprem/run.py"]
