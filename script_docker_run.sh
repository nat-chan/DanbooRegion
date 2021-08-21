#!/bin/bash

docker run --runtime=nvidia --rm -it -v /home:/home -v /data:/data region /bin/bash