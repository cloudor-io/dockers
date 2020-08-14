FROM nvidia/cuda:11.0-devel-ubuntu18.04

COPY ./cub/test /opt/cub/test
COPY ./cub/common.mk /opt/cub/common.mk

WORKDIR /opt/cub/test
RUN make -j 2 test_iterator sm=600,610,620,700
