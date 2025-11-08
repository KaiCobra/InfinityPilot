FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 安裝基本工具與 Miniconda
RUN apt-get update && apt-get install -y wget bzip2 git curl && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
 && bash miniconda.sh -b -p /opt/conda && rm miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# 複製 conda 環境設定檔
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml || true
ENV PATH=/opt/conda/envs/myenv/bin:$PATH

# 安裝與 V100 相容的 PyTorch（最新版仍支援 CUDA 11.8）
RUN pip install torch==2.5.0+cu118 torchvision==0.20.0+cu118 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

# 安裝 FlashAttention 1.x（V100 專用）
RUN pip install flash-attn==1.0.5.post0

WORKDIR /workspace
COPY . .

CMD ["/bin/bash"]
