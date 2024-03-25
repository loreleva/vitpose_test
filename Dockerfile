FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
RUN apt update && apt install git -y
RUN git clone https://github.com/loreleva/vitpose_test.git