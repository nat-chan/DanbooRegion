FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update\
    && apt-get install -y --no-install-recommends \
    wget gcc make zlib1g-dev libssl-dev libopencv-dev \
    curl git vim \
    libsqlite3-dev libreadline6-dev libbz2-dev libssl-dev libsqlite3-dev libncursesw5-dev libffi-dev libdb-dev libexpat1-dev zlib1g-dev liblzma-dev libgdbm-dev libmpdec-dev \
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
    && pip install -U h5py==2.10.0
RUN pip cache purge

# 以降JupyterLab3のインストール 
RUN curl -sL https://deb.nodesource.com/setup_16.x > /tmp/setup.sh
RUN bash /tmp/setup.sh
RUN rm /tmp/setup.sh
RUN apt-get update
RUN apt-get install -y --no-install-recommends nodejs

RUN pip install 'jupyterlab>=3.0.0,<4.0.0a0' jupyterlab-lsp
RUN pip install appmode
RUN jupyter nbextension     enable --py --sys-prefix appmode
RUN jupyter serverextension enable --py --sys-prefix appmode
RUN pip install ipycanvas orjson

RUN pip install jupyterlab_vim
RUN pip install ipyfilechooser
RUN pip install jupyterlab_widgets
RUN echo "jupyter lab --no-browser --port=8129 --ip=0.0.0.0 --allow-root --NotebookApp.token=''" > /lab.sh
RUN chmod +x /lab.sh
RUN pip install tqdm
RUN pip install pudb
#RUN jupyter labextension install jupyterlab-plotly

#RUN rm -rf /var/lib/apt/lists/*
#RUN pip cache purge