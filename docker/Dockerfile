ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 sox vim\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all
WORKDIR /flower-classification

ENV FORCE_CUDA="1"

RUN pip install cython==0.29.21
RUN pip install pandas==1.1.4
RUN pip install mmpycocotools==12.0.3
RUN pip install pycocotools==2.0.2
RUN pip install tensorflow-gpu==1.15
