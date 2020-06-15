FROM idock.daumkakao.io/kakaobrain/deepcloud-sshd:openpose-preprocess

COPY ./*.py /root/tf-openpose/
WORKDIR /root/tf-openpose/

RUN cd /root/tf-openpose/ && pip3 install -r requirements.txt

RUN cd core/tf_pose/pafprocess/ && swig -python -c++ pafprocess.i && python setup.py build_ext --inplace

ENTRYPOINT ["python3", "pose_dataworker.py"]
