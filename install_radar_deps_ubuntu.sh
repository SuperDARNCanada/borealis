#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
apt -yq update
apt -yq upgrade
apt-get install -yq wget
apt-get install -yq gcc
apt-get install -yq g++
apt-get install -yq vim 
apt-get install -yq git 
apt-get install -yq scons 
apt-get install -yq python-dev
apt-get install -yq python-pip 
pip2 install --upgrade pip
apt-get install -yq libx11-dev

#### INSTALL PROTOBUF ####i
#https://github.com/google/protobuf/blob/master/src/README.md#c-installation---uni 
apt-get install -yq autoconf automake libtool curl make unzip #protobuf deps
git clone https://github.com/google/protobuf.git
cd protobuf ||exit
./autogen.sh
./configure
make -j16
make -j16 check
make install
ldconfig # refresh shared library cache.
pip2 install protobuf
cd ../ || exit

#### INSTALL ZMQ ####
#https://github.com/zeromq/zmqpp#installation
# Build, check, and install libsodium
apt-get install -yq libboost-dev
git clone git://github.com/jedisct1/libsodium.git
cd libsodium || exit
git checkout stable
./autogen.sh 
./configure && make check 
make install 
ldconfig
cd ../ || exit

# Build, check, and install the latest version of ZeroMQ
git clone git://github.com/zeromq/libzmq.git
cd libzmq || exit
./autogen.sh 
./configure --with-libsodium && make -j16
make install
ldconfig
cd ../ || exit

# Now install C++ bindings
git clone https://github.com/zeromq/cppzmq.git
cd cppzmq || exit
cp zmq.hpp /usr/local/include/ 
cp zmq_addon.hpp /usr/local/include
cd ../ || exit
pip2 install zmq


#### INSTALL UHD ####
#http://files.ettus.com/manual/page_build_guide.html
apt-get install -yq libusb-1.0-0-dev python-mako doxygen python-docutils cmake 
git clone --recursive git://github.com/EttusResearch/uhd.git
cd uhd || exit
git submodule init
git submodule update
cd host || exit
mkdir build
cd build || exit
cmake ../
make -j16
make -j16 test
make install
ldconfig
cd ../../../ || exit

#### INSTALL CUDA ####
# https://developer.nvidia.com/cuda-toolkit
#kernel_version=`uname -r | awk -F'-' '{$NF=""; print $0}' | sed 's/\ /-/'`
#kernel_variant=`uname -r | awk -F'-' '{print $NF}'`
#apt-get install -yq kernel-${kernel_variant}-devel=${kernel_version}
apt-get install -yq linux-headers-generic
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
apt-get update
apt-get install -yq cuda
apt-get install -yq cuda # Seems to fail the first time due to 'no space left on device' error

