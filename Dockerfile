FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update\
    && apt-get install -y --no-install-recommends \
    wget gcc make zlib1g-dev libssl-dev libopencv-dev \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get remove -y --allow-change-held-packages libcudnn7


RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=7.0.5.15-1+cuda9.0 \
    && apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/3.6.11/Python-3.6.11.tgz \
    && tar zxvf Python-3.6.11.tgz \
    && cd Python-3.6.11 \
    && ./configure \
    && make && make install \
    && ln -s /usr/local/bin/python3.6 /usr/local/bin/python \
    && ln -s /usr/local/bin/pip3.6 /usr/local/bin/pip \
    && cd .. \
    && rm -r Python-3.6.11*

RUN pip install -U pip \
    && pip install tensorflow-gpu==1.5.0 \
    && pip install keras==2.2.4 \
    && pip install opencv-python==3.4.2.17 \
    && pip install numpy==1.15.4 \
    && pip install numba==0.49.0 \
    && pip install scipy==1.1.0 \
    && pip install scikit-image==0.13.0 \
    && pip install scikit-learn==0.22.2 \
    && pip install -U h5py==2.10.0 \
    && pip cache purge
