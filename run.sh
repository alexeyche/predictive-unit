#!/usr/bin/env bash

PU_SRC=$(python -c 'import os; print os.path.realpath(".")')

docker run \
   -u pu-user \
   -it \
   -p 8080:8080 \
   -v $PU_SRC:/predictive-unit-src \
   -v $HOME/.bash_aliases:/home/pu-user/.bash_aliases \
   -e DISPLAY=$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   pu \
   /bin/bash
