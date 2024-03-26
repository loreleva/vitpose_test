FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
RUN apt update && apt install git ffmpeg libsm6 libxext6 -y
RUN echo 4
RUN git clone https://github.com/loreleva/vitpose_test.git
WORKDIR ./vitpose_test
RUN pip install -r requirements.txt
WORKDIR ./easy_ViTPose
RUN pip install -e .
WORKDIR ..