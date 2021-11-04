nvidia-docker run -ti \
    -v $(pwd):/flower-classification/ \
    --ipc=host \
    --net=host \
    --name=flower-classification \
    flower-classification \
    /bin/bash

#    pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel \
