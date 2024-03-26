FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
RUN apt update && apt install git ffmpeg libsm6 libxext6  -y
RUN echo 2
RUN git clone https://github.com/loreleva/vitpose_test.git
WORKDIR ./vitpose_test
RUN pip install -r requirements.txt
WORKDIR ./easy_ViTPose
RUN pip install -e .
WORKDIR ..
#RUN pip install -r requirements.txt
#RUN pip install -r requirements_gpu.txt