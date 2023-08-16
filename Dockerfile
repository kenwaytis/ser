FROM tensorflow/tensorflow:2.13.0rc2-gpu
ARG HTTP_PROXY

ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTP_PROXY}
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home

RUN apt update && \
    apt install -y \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libbz2-dev \
    liblzma-dev \
    wget \
    ffmpeg \
    git && \
    apt upgrade -y gcc && \
    apt install -y \
    tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime &&\
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Jiaxin-Ye/TIM-Net_SER.git && \
    cd TIM-Net_SER/Code && \
    mkdir output && \
    mkdir test

WORKDIR /home/TIM-Net_SER/Code
COPY . .
RUN pip install -r requirement.txt


ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""