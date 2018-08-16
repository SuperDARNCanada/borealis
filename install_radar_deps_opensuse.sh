#!/bin/bash
zypper install -y wget
zypper install -y gcc
zypper install -y gcc-c++
zypper install -y vim 
zypper install -y git 
zypper install -y scons 
zypper install -y python-devel
zypper install -y python-pip 
pip2 install --upgrade pip
pip2 install scipy
pip2 install h5py
zypper install -y python-matplotlib
zypper install -y libX11-devel

#### INSTALL PROTOBUF ####i
#https://github.com/google/protobuf/blob/master/src/README.md#c-installation---uni 
zypper install -y autoconf automake libtool curl make unzip #protobuf deps
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
zypper install -y boost-devel
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
zypper install -y libusb-1_0-devel python-mako doxygen python-docutils cmake 
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
#zypper install -y kernel-${kernel_variant}-devel=${kernel_version}
zypper install -y kernel-devel
#wget http://developer.download.nvidia.com/compute/cuda/repos/opensuse422/x86_64/cuda-repo-opensuse422-9.0.176-1.x86_64.rpm
rpm -i cuda-repo-opensuse132-8.0.61-1.x86_64.rpm
zypper refresh
zypper install -y cuda
zypper install -y cuda # Seems to fail the first time due to 'no space left on device' error
