#!/bin/python3

"""
Copyright SuperDARN Canada 2020
Keith Kotyk

Installation script for Borealis utilities.
NOTE: This script has been tested on:
    OpenSuSe 15.1
    Ubuntu 19.10
    Ubuntu 20.04
"""
import subprocess as sp
import sys
import os
import multiprocessing as mp
import argparse as ap

def usage_msg():
    """
    Return the usage message for this process.

    This is used if a -h flag or invalid arguments are provided.

    :returns: the usage message
    """

    usage_message = """ install_radar_deps.py [-h] radar install_dir

    This script will download and configure all dependencies for the
    Borealis project. You must supply an installation directory.

    Example usage:
    python3 install_radar_deps.py sas /home/radar/borealis/

    """
    return usage_message

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
    # get Pretty name if available
    for line in os_info:
        if 'PRETTY_NAME' in line:
            distro = line.strip("PRETTY_NAME=").strip('"')
            break
    else:  # no break, no PRETTY_NAME
        distro = os_info[0].strip("NAME=").strip('"')

    return distro

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
                "screen",

                "hdf5",
                "libhdf5-dev",
                "hdf5-tools",

                "autoconf",
                "automake",
                "libtool",
                "curl",
                "make",
                "unzip",

                "libarmadillo9",
                "libarmadillo-dev",

                "pps-tools",

                "libboost_*_66_0",
                "libboost-*67.0*",  # Ubuntu 19.10
                "libboost-*71.0*",  # Ubuntu 20.04

                "python3-mako",
                "doxygen",
                "python3-docutils",
                "cmake",

                "uhd-udev",
                "libuhd-dev",

                "libgps23",
                "dpdk",

                "net-snmp-devel",
                "net-snmp-dev",
                "libsnmp-dev",

                "libevent-devel",
                "libevent-dev",

                "dpdk-devel",
                "dpdk-dev",

                "pps-tools-devel",
                "pps-tools-dev",
                "pps-tools",

                "libX11-devel",
                "libx11-dev",

                "python3-devel",
                "python3-dev",

                "python39",
                "python39-devel",

                "boost-devel",
                "liboost-all-dev",
                "libboost-dev",

                "libusb-1_0-devel",
                "libusb-1.0-0-dev",

                "dpdk-dev",
                "dpdk-devel",

                "kernel-devel",
                "linux-headers-generic",

                "unison",
                ]

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
        print(install_cmd)
        try:
            execute_cmd(install_cmd)
        except sp.CalledProcessError as e:
            print(e)

    update_pip = "pip3.9 install --upgrade pip"
    execute_cmd(update_pip)

    pip_cmd = "pip3.9 install " + " ".join(pip)
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
    "wget -N http://www.eecis.udel.edu/~ntp/ntp_spool/ntp4/ntp-4.2/ntp-4.2.8p13.tar.gz;" \
    "tar xvf ntp-4.2.8p13.tar.gz;" \
    "cd ntp-4.2.8p13/ || exit;" \
    "./configure --enable-atom;" \
    "make -j${CORES};" \
    "make install;"

    execute_cmd(ntp_cmd)

def install_uhd():
    """
    Install UHD. UHD is particular about which version of boost it uses, so check that.
    """


    def fix_boost_links():
        import glob
        import pathlib as pl
        libpath = ""

        if "openSUSE" in DISTRO:
            libpath = '/usr/lib64/'
            boost_version = "1.66"
        elif "Ubuntu" in DISTRO:
            if "20.04" in DISTRO:
                boost_version = "1.71"
            elif "19.10" in DISTRO:
                boost_version = "1.67"
            else:
                print("Ubuntu version {} unrecognized; exiting".format(DISTRO))
                sys.exit(1)
            libpath = '/usr/lib/x86_64-linux-gnu'
        else:
            print("Distro {} unrecognized; exiting".format(DISTRO))
            sys.exit(1)

        files = glob.glob('{}/libboost_*.so.{}*'.format(libpath, boost_version))
        print(files)

        files_with_no_ext = []

        for f in files:
            strip_ext = f
            while pl.Path(strip_ext).stem != strip_ext:
                strip_ext = pl.Path(strip_ext).stem
            files_with_no_ext.append(strip_ext)

        print(files_with_no_ext)

        for (f,n) in zip(files, files_with_no_ext):
            cmd = 'ln -s -f {} {}/{}.so'.format(f, libpath, n)
            execute_cmd(cmd)

        if "openSUSE" in DISTRO:
            cmd = 'ln -s -f {libpath}/libboost_python-py3.so {libpath}/libboost_python3.so'.format(libpath=libpath)
        elif "Ubuntu" in DISTRO:
            boost_python = glob.glob('{}/libboost_python3*.so.{}*'.format(libpath, boost_version))[0]
            cmd = 'ln -s -f {} {}/libboost_python3.so'.format(boost_python, libpath)
        execute_cmd(cmd)

    fix_boost_links()

    uhd_cmd = "cd ${IDIR};" \
    "git clone --recursive git://github.com/EttusResearch/uhd.git;" \
    "cd uhd || exit;" \
    "git checkout UHD-4.0;" \
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
    if "openSUSE" in DISTRO:
        cuda_file = 'cuda_10.2.89_440.33.01_linux.run'
    elif 'Ubuntu' in DISTRO:
        pre_cuda_setup_cmd = "apt-get install -y gcc-7 g++-7;" \
        "update-alternatives --remove-all gcc;" \
        "update-alternatives --remove-all g++;" \
        "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 50;" \
        "update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 50;" \
        "update-alternatives --config gcc;" \
        "update-alternatives --config g++;"
        execute_cmd(pre_cuda_setup_cmd)
        cuda_file = 'cuda_10.2.89_440.33.01_linux.run'

    cuda_cmd = "cd ${{IDIR}};" \
    "wget -N http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/{cuda_file};" \
    "sh {cuda_file} --silent --toolkit --samples;".format(cuda_file=cuda_file)

    execute_cmd(cuda_cmd)

def install_realtime():
    """
    Create virtual environment and install utilities needed for RT capabilities.
    """

    rt_cmd = "bash -c \"cd /usr/local;" \
    "git clone https://github.com/vtsuperdarn/hdw.dat.git;" \
    "mkdir -p $BOREALISPATH/borealisrt_env3.9;" \
    "virtualenv -p python3.9 $BOREALISPATH/borealisrt_env3.9;" \
    "source $BOREALISPATH/borealisrt_env3.9/bin/activate;" \
    "pip install zmq;" \
    "pip install git+git://github.com/SuperDARNCanada/backscatter.git#egg=backscatter;" \
    "pip install pydarnio;" \
    "deactivate;\""

    execute_cmd(rt_cmd)

def install_dspenv():
    """
    Create virtual environment and install utilities needed for python DSP.
    """

    rt_cmd = "bash -c \"mkdir -p $BOREALISPATH/dspenv;" \
    "virtualenv -p python3.9 $BOREALISPATH/dspenv;" \
    "source $BOREALISPATH/dspenv/bin/activate;" \
    "pip install zmq numpy scipy matplotlib cupy protobuf posix_ipc;" \
    "deactivate;\""

    execute_cmd(rt_cmd)

def install_directories():

    mkdirs_cmd = "mkdir -p /data/borealis_logs;" \
    "mkdir -p /data/borealis_data;" \
    "chown {normal_user}:{normal_group} /data/borealis_*;"

    mkdirs_cmd = mkdirs_cmd.format(normal_user=args.user, normal_group=args.group)

    execute_cmd(mkdirs_cmd)

def install_hdw_dat():

    install_hdw_cmd = "cp -v /usr/local/hdw.dat/hdw.dat.{radar_abbreviation} $BOREALISPATH"
    install_hdw_cmd = install_hdw_cmd.format(radar_abbreviation=args.radar)

def install_config():

    install_config_cmd = "bash -c 'cd $BOREALISPATH'; git submodule update --init;" \
    "chown -R {normal_user}:{normal_group} borealis_config_files;"
    install_config_cmd = install_config_cmd.format(normal_user=args.user, normal_group=args.group)
    execute_cmd(install_config_cmd)


parser = ap.ArgumentParser(usage=usage_msg(), description="Installation script for Borealis utils")
parser.add_argument("--borealis-dir", help="Path to the Borealis installation directory", default="")
parser.add_argument("--user", help="The username of the user who will run borealis. Default 'radar'", default="radar")
parser.add_argument("--group", help="The group name of the user who will run borealis. Default 'users'", default="users")
parser.add_argument("radar", help="The three letter abbreviation for this radar. Example: sas")
parser.add_argument("install_dir", help="Path to the installation directory")
args = parser.parse_args()

if os.geteuid() != 0:
    print("You must run this script as root.")
    sys.exit(1)

if not os.path.isdir(args.install_dir):
    print("Install directory does not exist: {}".format(args.install_dir))
    sys.exit(1)

if args.borealis_dir == "":
    try:
        BOREALISPATH = os.environ['BOREALISPATH']
    except KeyError as e:
        print("You must have an environment variable set for BOREALISPATH.")
        sys.exit(1)
else:
    os.environ['BOREALISPATH'] = args.borealis_dir

DISTRO = get_distribution()

# Set env variables that will be read by subshells
os.environ['IDIR'] = args.install_dir
os.environ['CORES'] = str(mp.cpu_count())

install_packages()
install_protobuf()
install_zmq()
install_ntp()
install_uhd()
install_cuda()
install_realtime()
install_dspenv()
install_directories()
install_hdw_dat()
install_config()
