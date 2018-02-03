FROM ubuntu

ENV LD_LIBRARY_PATH "/usr/local/lib"

ADD https://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2 /contrib/eigen.tar.bz2
ADD https://github.com/google/protobuf/archive/v3.5.1.tar.gz /contrib/protobuf.tar.gz
ADD https://github.com/pocoproject/poco/archive/poco-1.8.1-release.tar.gz /contrib/poco.tar.gz

WORKDIR /contrib

RUN apt-get update
RUN apt-get install -y \
	build-essential \
	gdb \
	cmake \
	bzip2 \
	unzip \
	autoconf \
	libtool \
	golang \
	libspdlog-dev \
	libblas-dev \
	liblapack-dev \
	git \
	openssl \
	libssl-dev \
	libiodbc2 \
	libiodbc2-dev \
	curl \
	vim \
	sudo \
	python \
	python-pip \ 
	telnet \
	netcat

RUN pip install protobuf

RUN mkdir eigen && tar -xf eigen.tar.bz2 -C eigen; \
    cd eigen; \
	cmake ./eigen*; \
	make -j6 install

RUN git clone https://github.com/jbeder/yaml-cpp
RUN cd yaml-cpp; \
	git checkout 86ae3a5aa7e2109d849b2df89176d6432a35265d; \
	cmake -DBUILD_SHARED_LIBS=ON .; \
	make -j6 install

RUN tar -xf /contrib/protobuf.tar.gz -C .; \
	cd protobuf-*; \
 	./autogen.sh; \
 	./configure --enable-shared; \ 
	make -j6 install

RUN tar -xf /contrib/poco.tar.gz -C .; \
	cd poco-*; \
	mkdir poco_build; \
	cd poco_build; \
	cmake ..; \
	make -j6 install

