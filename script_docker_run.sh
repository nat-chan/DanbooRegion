#!/bin/bash

docker run \
--runtime=nvidia \
--rm -it \
-v /home:/home \
-v /data:/data \
-w $PWD \
region /bin/bash
#-u $UID:$GID \
#-p 8219:8129 \