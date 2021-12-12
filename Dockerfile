ARG AIDO_REGISTRY

FROM nvidia/opengl:1.2-glvnd-devel

RUN apt-get update -y && apt-get install -y  \
    freeglut3-dev \
    python3-pip \
    python3-numpy \
    python3-scipy \
    xvfb \
    wget curl vim git \
    && \
    rm -rf /var/lib/apt/lists/*

ADD gym-duckietown .

RUN pip install --upgrade pip
RUN pip install -e .

RUN apt-get install -y git
RUN git clone https://github.com/Balint197/DeepDriveDucks.git

RUN pip install pyvirtualdisplay
RUN pip install pyglet==1.5.15

# deepdriveducks dependecies =============================================

RUN pip install -U ray[tune]  # installs Ray + dependencies for Ray Tune
RUN pip install -U ray[rllib]  # installs Ray + dependencies for Ray RLlib

RUN apt update
RUN apt-get install python-opengl -y
RUN apt-get install x11-utils -y
RUN pip install tqdm
RUN pip install torchvision
RUN pip install tensorboard
RUN pip install piglet
RUN pip install pyvirtualdisplay

# WORKDIR DeepDriveDucks

# ENTRYPOINT [ "xvfb-run" ]
