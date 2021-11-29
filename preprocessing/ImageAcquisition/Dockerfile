FROM python:3.9

MAINTAINER Andrea Napoletani <andrea.napoletani@gmail.com>

RUN mkdir /images
WORKDIR /app
RUN apt-get update && apt-get install -y \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    nano

RUN pip3 install --upgrade pip && \
    pip3 install keyboard \
    pip install opencv-contrib-python \
    pip install numpy

COPY ImageAcquisition /app
CMD [ "python3", "split.py" ]
