FROM ubuntu

ADD https://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2 /contrib/eigen.tar.bz2

WORKDIR /contrib

RUN apt-get update
RUN apt-get install -y bzip2
RUN apt-get install -y build-essential cmake
RUN apt-get install -y libblas-dev liblapack-dev
RUN mkdir eigen && tar -xf eigen.tar.bz2 -C eigen
RUN cd eigen; cmake ./eigen*; make install; cd ..

RUN apt-get install -y git
RUN git clone https://github.com/jbeder/yaml-cpp
RUN cd yaml-cpp; git checkout 86ae3a5aa7e2109d849b2df89176d6432a35265d; cmake -DBUILD_SHARED_LIBS=ON .; make -j install

ENV LD_LIBRARY_PATH "/usr/local/lib"

RUN apt-get install -y libspdlog-dev
RUN apt-get install -y gdb

RUN apt-get install -y autoconf libtool
RUN git clone -b v1.8.x https://github.com/grpc/grpc

RUN cd grpc; git submodule update --init

RUN apt-get install -y golang curl unzip

RUN cd grpc/third_party/protobuf; \
 	./autogen.sh; \
 	./configure --enable-shared; \ 
	make -j8 install

RUN mkdir grpc-build; cd grpc-build; cmake -DgRPC_PROTOBUF_PROVIDER=package /contrib/grpc; make -j8 install
