FROM nvidia/cuda:12.9.0-devel-ubuntu22.04


RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && apt-get install -y \
    gcc-10 g++-10

ENV CC=/usr/bin/gcc-10
ENV CXX=/usr/bin/g++-10
ENV CUDAHOSTCXX=/usr/bin/g++-10
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


RUN apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    libcurl4-openssl-dev \
    libopenblas-dev \
    libcgal-qt5-dev \
    python3-pip \
    python3-venv \
    ssh \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*


RUN git clone --recursive https://github.com/colmap/colmap.git /colmap && \
    cd /colmap && \
    git checkout tags/3.12.3 -b v3.12.3 && \
    mkdir /colmap/build && cd /colmap/build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=86 && \
    ninja && \
    ninja install


RUN mkdir -p /colmap/scripts/python && \
    wget -O /colmap/scripts/python/colmap2nerf.py https://raw.githubusercontent.com/NVlabs/instant-ngp/refs/heads/master/scripts/colmap2nerf.py


RUN pip3 install --upgrade pip && \
    pip3 install numpy opencv-python

RUN pip3 install pillow

RUN pip3 install albumentations

#SAM & TORCH venv cause numpy<2.0
RUN python3 -m venv /venv_sr && \
    /venv_sr/bin/pip install --upgrade pip && \
    /venv_sr/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.2.0+cu121 torchvision==0.17.0+cu121 einops && \
    /venv_sr/bin/pip install "numpy<2" pillow tqdm opencv-python matplotlib && \
    /venv_sr/bin/pip install scikit-learn timm && \
    /venv_sr/bin/pip install git+https://github.com/facebookresearch/segment-anything.git
RUN mkdir -p /app/checkpoints && \
    wget -O /app/checkpoints/sam_vit_b.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

WORKDIR /app

COPY fix_relative_img_paths.py .
COPY augment_dataset.py .
COPY entrypoint.sh .
COPY downscale.py /app/downscale.py
COPY no_background_sam.py /app/no_background_sam.py
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
