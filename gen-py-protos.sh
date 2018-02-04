#!/usr/bin/env bash

set -ex

PU_SRC=$(python -c 'import os; print os.path.realpath(".")')

DST_PROTOS=${1:-/var/tmp/predictive-unit-protos}
mkdir -p $DST_PROTOS

SCRIPT=$(cat <<-END
protoc \
     -I /predictive-unit-src/predictive-unit/protos \
     --python_out /dst-protos \
     /predictive-unit-src/predictive-unit/protos/*.proto
END
)

docker run \
    -u $(id -u) \
   -it \
   -v $PU_SRC:/predictive-unit-src \
   -v $DST_PROTOS:/dst-protos \
   pu \
   /bin/bash -c "$SCRIPT"
   
rm -f $DST_PROTOS/*.pyc   