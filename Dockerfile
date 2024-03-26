FROM dustynv/pytorch:2.1-r36.2.0
#RUN apt update && apt install git ffmpeg libsm6 libxext6 -y
RUN echo 4
RUN git clone https://github.com/loreleva/vitpose_test.git
WORKDIR ./vitpose_test
RUN pip install -r requirements.txt
RUN pip install -e ./easy_ViTPose


#pip install "opencv-python-headless<4.3"