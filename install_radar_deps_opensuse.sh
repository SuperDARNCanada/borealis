#!/bin/bash

CORES=$(grep ^cpu\\scores /proc/cpuinfo | wc -l)

zypper install -y wget
zypper install -y gcc
zypper install -y gcc-c++
zypper install -y vim
zypper install -y git
zypper install -y scons
zypper install -y python3-devel
zypper install -y python3-pip
zypper install -y gdb
zypper install -y jq
zypper install -y hdf5
pip3 install --upgrade pip
zypper install -y libX11-devel
pip3 install deepdish
pip3 install posix_ipc
pip3 install inotify
pip3 install matplotlib
pip3 install virtualenv
pip3 install Sphinx
pip3 install sphinxcontrib-programoutput
pip3 install sphinxcontrib-autoprogram
pip3 install breathe

#### INSTALL PROTOBUF ####
#https://github.com/google/protobuf/blob/master/src/README.md#c-installation---uni
zypper install -y autoconf automake libtool curl make unzip #protobuf deps
git clone https://github.com/google/protobuf.git
cd protobuf ||exit
./autogen.sh
./configure
make -j${CORES}
make -j${CORES} check
make install
ldconfig # refresh shared library cache.
pip3 install protobuf
cd ../ || exit

#### INSTALL ZMQ ####
#https://github.com/zeromq/zmqpp#installation
# Build, check, and install libsodium
git clone git://github.com/jedisct1/libsodium.git
cd libsodium || exit
git checkout stable
./autogen.sh
./configure && make -j${CORES} check
make install
ldconfig
cd ../ || exit

# Build, check, and install the latest version of ZeroMQ
git clone git://github.com/zeromq/libzmq.git
cd libzmq || exit
./autogen.sh
./configure --with-libsodium && make -j${CORES}
make install
ldconfig
cd ../ || exit

# Now install C++ bindings
git clone https://github.com/zeromq/cppzmq.git
cd cppzmq || exit
cp zmq.hpp /usr/local/include/
cp zmq_addon.hpp /usr/local/include
cd ../ || exit
pip3 install zmq

#### INSTALL ARMADILLO ####
zypper install -y libarmadillo9 armadillo-devel

#### INSTALL NTPD with PPS support ####
zypper install -y pps-tools pps-tools-devel
cp -v /usr/include/sys/timepps.h /usr/include/ || exit
#modprobe pps_ldisc <- can happen in startup script
wget http://www.eecis.udel.edu/~ntp/ntp_spool/ntp4/ntp-4.2/ntp-4.2.8p13.tar.gz
tar xvf ntp-4.2.8p13.tar.gz
cd ntp-4.2.8p13/ || exit
./configure --enable-atom
make -j${CORES}
make install
cd ../ || exit
# ldattach 18 /dev/ttyS0 <- startup script
# /usr/local/bin/ntpd <- startup script

#### INSTALL BOOST ####
zypper install -y boost-devel libboost_*_66_0
#this embedded python piece fixes Opensuse boost links
python3 -<<END
import glob
import os
import subprocess as sp
import pathlib as pl

files = glob.glob('/usr/lib64/libboost_*')
print(files)

files_with_no_ext = []

for f in files:
    strip_ext = f
    while pl.Path(strip_ext).stem != strip_ext:
        strip_ext = pl.Path(strip_ext).stem
    files_with_no_ext.append(strip_ext)

print(files_with_no_ext)

for (f,n) in zip(files, files_with_no_ext):
    cmd = 'ln -s {} /usr/lib64/{}.so'.format(f,n)
    sp.call(cmd.split())

cmd = 'ln -s /usr/lib64/libboost_python-py3.so /usr/lib64/libboost_python3.so'
sp.call(cmd.split())
END

#### INSTALL UHD ####
#http://files.ettus.com/manual/page_build_guide.html
zypper install -y libusb-1_0-devel python3-mako doxygen python3-docutils cmake uhd-udev libgps23 dpdk dpdk-devel
git clone --recursive git://github.com/EttusResearch/uhd.git
cd uhd || exit
git checkout UHD-3.14
git submodule init
git submodule update
cd host || exit
mkdir build
cd build || exit
cmake -DENABLE_PYTHON3=on -DPYTHON_EXECUTABLE=$(which python3) -DRUNTIME_PYTHON_EXECUTABLE=$(which python3) -DENABLE_PYTHON_API=ON -DENABLE_DPDK=OFF ../ #DPDK would be nice to have in later revisions if its added to N200
make -j${CORES}
make -j${CORES} test
make install
ldconfig
cd ../../../ || exit

#### INSTALL CUDA ####
zypper install -y kernel-devel
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
sh cuda_10.1.243_418.87.00_linux.run --silent --toolkit --samples 

### INSTALL PYDARN FOR REALTIME AND TESTING ###
cd $HOME
git clone https://github.com/SuperDARN/pydarn.git 

#### REALTIME ####
cd /usr/local
git clone https://github.com/vtsuperdarn/hdw.dat.git
mkdir $BOREALISPATH/borealisrt_env
virtualenv $BOREALISPATH/borealisrt_env
source $BOREALISPATH/borealisrt_env/bin/activate
pip install zmq
pip install git+git://github.com/SuperDARNCanada/backscatter.git#egg=backscatter
cd $HOME/pydarn
git checkout develop
python setup.py install
deactivate

### TESTING AND DATA CONVERSIONS PACKAGES ###
cd $HOME
git clone https://github.com/SuperDARNCanada/borealis-data-utils.git 
git clone https://github.com/SuperDARNCanada/data_flow.git
mkdir $HOME/pydarn-env
virtualenv $HOME/pydarn-env
source $HOME/pydarn-env/bin/activate
cd $HOME/pydarn
git checkout rc_v1.0.0
python3 setup.py install
deactivate
