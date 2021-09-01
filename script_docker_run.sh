#!/bin/bash

docker run \
--runtime=nvidia \
--rm -it \
-v /home:/home \
-v /data:/data \
-p 8219:8129 \
region /bin/bash