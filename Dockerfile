FROM nvcr.io/nvidia/pytorch:25.01-py3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND="noninteractive"

COPY requirements.txt requirements.txt
RUN apt update \
    && apt install -y unzip python3-pip python3-setuptools sudo openjdk-21-jdk openjdk-21-jre \
    && pip install -r requirements.txt

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"
RUN pip install git+https://github.com/facebookresearch/xformers.git@v0.0.29.post3 --no-deps