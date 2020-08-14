#!/bin/bash

git clone https://github.com/NVlabs/cub.git
docker pull nvidia/cuda:11.0-devel-ubuntu18.04
docker build . -t cloudorio/cub_test:11.0-devel-ubuntu18.04

docker pull nvidia/cuda:11.0-runtime-ubuntu18.04

# copy the binary out
docker create -ti --name temp cloudorio/cub_test:11.0-devel-ubuntu18.04 bash
docker cp temp:/opt/cub/test ./test
docker rm -f temp

docker build . -t cloudorio/cub_test:11.0-runtime-ubuntu18.04 -f Dockerfile_runtime
docker push cloudorio/cub_test:11.0-runtime-ubuntu18.04
