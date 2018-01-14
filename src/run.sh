#!/usr/bin/env bash

PU_SRC=$(python -c 'import os; print os.path.realpath(".")')

docker run \
   -u pu-user \
   -it \
   -v $PU_SRC:/predictive-unit-src \
   -v $HOME/.bash_aliases:/home/pu-user/.bash_aliases \
   pu \
   /bin/bash
