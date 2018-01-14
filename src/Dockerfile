FROM pu-base

RUN useradd -ms /bin/bash pu-user

WORKDIR /home/pu-user/predictive-unit-build
RUN chown pu-user /home/pu-user/predictive-unit-build
RUN chgrp pu-user /home/pu-user/predictive-unit-build

USER pu-user

ARG PU_PATH 
ENV PU_PATH ${PU_PATH:-/usr/share/predictive-unit}

ARG PUB_PATH 
ENV PUB_PATH ${PUB_PATH:-/usr/share/predictive-unit-build}


#RUN mkdir $SMB_PATH; \
#	cd $SMB_PATH; \
#    cmake $SM_PATH; \
#    make -j
#
#RUN ln -s /usr/share/sparse-map-build/test-sparse-map /

