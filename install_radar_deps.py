#!/bin/python3

"""
Copyright SuperDARN Canada 2020
Keith Kotyk

Installation script for Borealis utilities.

"""
import subprocess as sp
import sys
import os
import multiprocessing as mp

def execute_cmd(cmd):
    """
    Execute a shell command and return the output

    :param      cmd:  The command
    :type       cmd:  string

    :returns:   Decoded output of the command.
    :rtype:     string
    """
    output = sp.check_output(cmd, shell=True)
    return output.decode('utf-8')

def get_distribution():
    """
    Gets the linux distribution type.

    :returns:   The distribution name.
    :rtype:     str
    """

    os_info = execute_cmd("cat /etc/os-release")

    os_info = os_info.splitlines()
    distro = os_info[0].strip("NAME=").strip('"')

    return distro

DISTRO = get_distribution()

# Set env variables that will be read by subshells
os.environ['IDIR'] = sys.argv[1]
os.environ['CORES'] = str(mp.cpu_count())


def install_packages():
    """
    Install the needed packages used by Borealis. Multiple options are listed for distributions that
    use different names.
    """

    packages = ["wget",
                "gcc",
                "gcc-c++",
                "vim",
                "git",
                "scons",
                "python3-pip",
                "gdb",
                "jq",
                "hdf5",
                "autoconf",
                "automake",
                "libtool",
                "curl",
                "make",
                "unzip",
                "libarmadillo9",
                "pps-tools",
                "libboost_*_66_0",
                "python3-mako",
                "doxygen",
                "python3-docutils",
                "cmake",
                "uhd-udev",
                "libgps23",
                "dpdk",

                "net-snmp-devel",
                "net-snmp-dev",

                "libevent-devel",
                "libevent-dev",

                "dpdk-devel",
                "dpdk-dev",

                "pps-tools-devel",
                "pps-tools-dev",

                "libX11-devel",
                "libX11-dev",

                "python3-devel",
                "python3-dev",

                "boost-devel",
                "liboost-all-dev",

                "libusb-1_0-devel",
                "libusb-1.0-0-dev",

                "dpdk-dev",
                "dpdk-devel",

                "kernel-devel",
                "linux-headers-generic"]

    pip = ["deepdish",
            "posix_ipc",
            "inotify",
            "matplotlib",
            "virtualenv",
            "protobuf",
            "zmq"]

    if "openSUSE" in DISTRO:
        pck_mgr = 'zypper'
    elif 'Ubuntu' in DISTRO:
        pck_mgr = 'apt-get'
    else:
        print("Could not detect package manager type")
        sys.exit(-1)

    for pck in packages:
        install_cmd = pck_mgr + " install -y " + pck
        try:
            execute_cmd(install_cmd)
        except sp.CalledProcessError as e:
            print(e)


    update_pip = "pip3 install --upgrade pip"
    execute_cmd(update_pip)

    pip_cmd = "pip3 install " + " ".join(pip)
    execute_cmd(pip_cmd)


def install_protobuf():
    """
    Install protobuf.
    """

    proto_cmd = "cd ${IDIR};" \
    "git clone https://github.com/google/protobuf.git;" \
    "cd protobuf || exit;" \
    "./autogen.sh;" \
    "./configure;" \
    "make -j${CORES};" \
    "make -j${CORES} check;" \
    "make install;" \
    "ldconfig;"

    execute_cmd(proto_cmd)

def install_zmq():
    """
    Install ZMQ and C++ bindings.
    """

    zmq_cmd = "cd ${IDIR};" \
    "git clone git://github.com/jedisct1/libsodium.git;" \
    "cd libsodium || exit;" \
    "git checkout stable;" \
    "./autogen.sh;" \
    "./configure && make -j${CORES} check;" \
    "make install;" \
    "ldconfig;" \
    "cd ../ || exit;" \
    "git clone git://github.com/zeromq/libzmq.git;" \
    "cd libzmq || exit;" \
    "./autogen.sh;" \
    "./configure --with-libsodium && make -j${CORES};" \
    "make install;" \
    "ldconfig;" \
    "cd ../ || exit;" \
    "git clone https://github.com/zeromq/cppzmq.git;" \
    "cd cppzmq || exit;" \
    "cp zmq.hpp /usr/local/include/;" \
    "cp zmq_addon.hpp /usr/local/include;"

    execute_cmd(zmq_cmd)

def install_ntp():
    """
    Install NTP with PPS support.
    """

    ntp_cmd = "cd ${IDIR};" \
    "cp -v /usr/include/sys/timepps.h /usr/include/ || exit;" \
    "wget http://www.eecis.udel.edu/~ntp/ntp_spool/ntp4/ntp-4.2/ntp-4.2.8p13.tar.gz;" \
    "tar xvf ntp-4.2.8p13.tar.gz;" \
    "cd ntp-4.2.8p13/ || exit;" \
    "./configure --enable-atom;" \
    "make -j${CORES};" \
    "make install;"

    execute_cmd(ntp_cmd)

def install_uhd():
    """
    Install UHD.
    """


    def fix_boost_links():
        import glob
        import pathlib as pl

        files = glob.glob('/usr/lib64/libboost_*.so.*')
        print(files)

        files_with_no_ext = []

        for f in files:
            strip_ext = f
            while pl.Path(strip_ext).stem != strip_ext:
                strip_ext = pl.Path(strip_ext).stem
            files_with_no_ext.append(strip_ext)

        print(files_with_no_ext)

        for (f,n) in zip(files, files_with_no_ext):
            cmd = 'ln -s -f {} /usr/lib64/{}.so'.format(f,n)
            execute_cmd(cmd)

        cmd = 'ln -s -f /usr/lib64/libboost_python-py3.so /usr/lib64/libboost_python3.so'
        execute_cmd(cmd)

    if "openSUSE" in DISTRO:
        fix_boost_links()

    uhd_cmd = "cd ${IDIR};" \
    "git clone --recursive git://github.com/EttusResearch/uhd.git;" \
    "cd uhd || exit;" \
    "git checkout UHD-3.14;" \
    "git submodule init;" \
    "git submodule update;" \
    "cd host || exit;" \
    "mkdir build;" \
    "cd build || exit;" \
    "cmake -DENABLE_PYTHON3=on -DPYTHON_EXECUTABLE=$(which python3) -DRUNTIME_PYTHON_EXECUTABLE=$(which python3) -DENABLE_PYTHON_API=ON -DENABLE_DPDK=OFF ../;" \
    "make -j${CORES};" \
    "make -j${CORES} test;" \
    "make install;" \
    "ldconfig;"

    execute_cmd(uhd_cmd)

def install_cuda():
    """
    Install CUDA.
    """

    cuda_cmd = "cd ${IDIR};" \
    "wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run;" \
    "sh cuda_10.1.243_418.87.00_linux.run --silent --toolkit --samples;"

    execute_cmd(cuda_cmd)

def install_realtime():
    """
    Create virtual environment and install utilities needed for RT capabilities.
    """

    rt_cmd = "bash -c \"cd /usr/local;" \
    "git clone https://github.com/vtsuperdarn/hdw.dat.git;" \
    "mkdir $BOREALISPATH/borealisrt_env;" \
    "virtualenv $BOREALISPATH/borealisrt_env;" \
    "source $BOREALISPATH/borealisrt_env/bin/activate;" \
    "pip install zmq;" \
    "pip install git+git://github.com/SuperDARNCanada/backscatter.git#egg=backscatter;" \
    "pip install pydarn;" \
    "deactivate;\""

    execute_cmd(rt_cmd)

install_packages()
install_protobuf()
install_zmq()
install_ntp()
install_uhd()
install_cuda()
install_realtime()
