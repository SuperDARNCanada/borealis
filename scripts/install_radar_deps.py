#!/usr/bin/env python3

"""
    install_radar_deps
    ~~~~~~~~~~~~~~~~~~
    Installation script for Borealis utilities.
    NOTE: This script has been tested on:
        OpenSuSe 15.1-15.3
        Ubuntu 19.10
        Ubuntu 20.04

    :copyright: 2020 SuperDARN Canada
    :author: Keith Kotyk
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

    :returns:   the usage message
    :rtype:     str
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
    :type       cmd:  str

    :returns:   Decoded output of the command.
    :rtype:     str
    """
    # try/except block lets install script continue even if something fails
    try:
        output = sp.check_output(cmd, shell=True)
    except sp.CalledProcessError as err:
        output = err.output

    output = output.decode('utf-8')
    if len(output) > 0:     # Don't print new lines if there was no output
        print(output)  # catches echo statements
    return output


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


def install_python(distro: str, python_version: str):
    """
    Install the required Python packages for the specified version.

    :param distro:          Distribution to install on
    :type  distro:          str
    :param python_version:  Version of python to install
    :type  python_version:  str
    """
    print(f'### Installing python{python_version} ###')
    if 'openSUSE' not in distro:
        raise ValueError(f'ERROR: Unable to install python for {distro}. Please consult the documentation for {distro} '
                         f'for instructions on how to install python{python_version}')
    else:
        short_version = ''.join(python_version.split('.'))
        packages = [
            'python3-devel',
            'python3-pip',
            f'python{short_version}',
            f'python{short_version}-devel',
        ]
        for pck in packages:
            install_cmd = f'zypper install -y {pck}'
            print(install_cmd)
            execute_cmd(install_cmd)


def install_packages(distro: str):
    """
    Install the needed packages used by Borealis. Multiple options are listed for distributions that
    use different names.

    :param  distro:         Distribution to install on
    :type   distro:         str
    """
    print('### Installing system packages ###')

    common_packages = [
        "gcc",
        "gcc-c++",
        "git",
        "scons",                # For building Borealis
        "gdb",                  # For engineeringdebug mode
        "jq",                   # For reading JSON files from the command line
        "at",                   # Required for the scheduler
        "mutt",                 # Required for the scheduler
        "hdf5",
        "autoconf",             # Required for installing protobuf and zmq
        "automake",             # Required for installing protobuf and zmq
        "libtool",              # Required for installing protobuf and zmq
        "python3-mako",         # Required for UHD
        "cmake",
        "pps-tools",
        ]

    variant_packages = [
        "libusb-1_0-{}",    # Needed for UHD
        "libX11-{}",
        "pps-tools-{}",
        "net-snmp-{}",
    ]

    ubuntu_packages = [
        "libsnmp-dev",
        "libusb-1.0-0-dev",
        "libhdf5-dev",
        "liboost-all-dev",
        "libboost-dev",
        "hdf5-tools",
        "linux-headers-generic",
        "libboost-*67.0*",  # Ubuntu 19.10
        "libboost-*71.0*",  # Ubuntu 20.04
    ]

    opensuse_packages = [
        "kernel-devel",
        "libboost_*_66_0",
        "uhd-devel",
    ]

    if "openSUSE" in distro:
        pck_mgr = 'zypper'
        variant_packages = [pck.format('devel') for pck in variant_packages]
        all_packages = common_packages + variant_packages + opensuse_packages
    elif 'Ubuntu' in distro:
        pck_mgr = 'apt-get'
        variant_packages = [pck.format('dev') for pck in variant_packages]
        all_packages = common_packages + variant_packages + ubuntu_packages
    else:
        print("Could not detect package manager type")
        sys.exit(-1)

    for pck in all_packages:
        install_cmd = pck_mgr + " install -y " + pck
        print(install_cmd)
        execute_cmd(install_cmd)


def pip_install_packages(user: str, python_version: str):
    """
    Install required python packages using pip

    :param user:            User to install as
    :type  user:            str
    :param python_version:  Version of python to install on
    :type  python_version:  str
    """
    packages = [
        'wheel',
        'deepdish',
        'posix_ipc',
        'inotify',
        'matplotlib',
        'virtualenv',
        'protobuf==3.19.4',
        'numpy',
        'zmq',
    ]
    update_pip = f'sudo -u {user} python{python_version} -m pip install --upgrade pip'
    print(update_pip)
    execute_cmd(update_pip)

    pip_cmd = f'sudo -u {user} python{python_version} -m pip install ' + ' '.join(packages)
    print(pip_cmd)
    execute_cmd(pip_cmd)


def install_protobuf():
    """
    Install protobuf.
    """
    print('### Installing protocol buffers ###')
    proto_cmd = "cd ${IDIR};" \
                "git clone https://github.com/protocolbuffers/protobuf.git;" \
                "cd protobuf || exit;" \
                "git checkout v3.19.4;" \
                "git submodule init && git submodule update;" \
                "./autogen.sh;" \
                "./configure;" \
                "make;" \
                "make check;" \
                "make install;" \
                "ldconfig;"

    execute_cmd(proto_cmd)


def install_zmq():
    """
    Install ZMQ and C++ bindings.
    """
    print('### Installing ZeroMQ ###')
    libsodium_cmd = "cd ${IDIR};" \
                    "wget https://download.libsodium.org/libsodium/releases/LATEST.tar.gz;" \
                    "tar xzf LATEST.tar.gz;" \
                    "cd libsodium-stable || exit;" \
                    "./configure;" \
                    "make -j${CORES} && make -j${CORES} check;" \
                    "make install;" \
                    "ldconfig;"
    execute_cmd(libsodium_cmd)

    zmq_cmd = "cd ${IDIR};" \
              "git clone https://github.com/zeromq/libzmq.git;" \
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
    print('### Installing NTP ###')
    ntp_cmd = "cd ${IDIR};" \
              "cp -v /usr/include/sys/timepps.h /usr/include/ || exit;" \
              "wget -N http://www.eecis.udel.edu/~ntp/ntp_spool/ntp4/ntp-4.2/ntp-4.2.8p13.tar.gz;" \
              "tar xvf ntp-4.2.8p13.tar.gz;" \
              "cd ntp-4.2.8p13/ || exit;" \
              "./configure --enable-atom;" \
              "make -j${CORES};" \
              "make install;"

    execute_cmd(ntp_cmd)


def install_uhd(distro: str):
    """
    Install UHD. UHD is particular about which version of boost it uses, so check that.

    :param  distro: Distribution to install on
    :type   distro: str
    """
    if 'openSUSE' in distro:    # Already installed via zypper
        return

    print('### Installing UHD ###')

    def fix_boost_links():
        import glob
        import pathlib as pl

        if "openSUSE" in distro:
            libpath = '/usr/lib64/'
            boost_version = "1.66"
        elif "Ubuntu" in distro:
            if "20.04" in distro:
                boost_version = "1.71"
            elif "19.10" in distro:
                boost_version = "1.67"
            else:
                print(f"Ubuntu version {distro} unrecognized; exiting")
                sys.exit(1)
            libpath = '/usr/lib/x86_64-linux-gnu'
        else:
            print(f"Distro {distro} unrecognized; exiting")
            sys.exit(1)

        files = glob.glob(f'{libpath}/libboost_*.so.{boost_version}*')
        print(files)

        files_with_no_ext = []

        for f in files:
            strip_ext = f
            while pl.Path(strip_ext).stem != strip_ext:
                strip_ext = pl.Path(strip_ext).stem
            files_with_no_ext.append(strip_ext)

        print(files_with_no_ext)

        for (f, n) in zip(files, files_with_no_ext):
            cmd = f'ln -s -f {f} {libpath}/{n}.so'
            execute_cmd(cmd)

        cmd = ""
        if "openSUSE" in distro:
            cmd = f'ln -s -f {libpath}/libboost_python-py3.so {libpath}/libboost_python3.so'
        elif "Ubuntu" in distro:
            boost_python = glob.glob(f'{libpath}/libboost_python3*.so.{boost_version}*')[0]
            cmd = f'ln -s -f {boost_python} {libpath}/libboost_python3.so'
        execute_cmd(cmd)

    fix_boost_links()

    uhd_cmd = "cd ${IDIR};" \
              "git clone --recursive https://github.com/EttusResearch/uhd.git;" \
              "cd uhd || exit;" \
              "git checkout UHD-4.4;" \
              "git submodule init;" \
              "git submodule update;" \
              "cd host || exit;" \
              "mkdir build;" \
              "cd build || exit;" \
              "cmake -DENABLE_PYTHON3=on -DPYTHON_EXECUTABLE=$(which python3) " \
              "-DRUNTIME_PYTHON_EXECUTABLE=$(which python3) -DENABLE_PYTHON_API=ON -DENABLE_DPDK=OFF ../;" \
              "make -j${CORES};" \
              "make -j${CORES} test;" \
              "make install;" \
              "ldconfig;"

    execute_cmd(uhd_cmd)


def install_cuda(distro: str):
    """
    Install CUDA.
    
    :param  distro: Distribution to install on
    :type   distro: str
    """
    print('### Installing CUDA ###')

    if "openSUSE" in distro:
        pre_cuda_setup_cmd = "groupadd video;" \
                             "usermod -a -G video $USER;" \
                             "rpm --erase gpg-pubkey-7fa2af80*"
        execute_cmd(pre_cuda_setup_cmd)
        cuda_zypper_cmd = \
            "zypper removerepo cuda-opensuse15-x86_64;" \
            "zypper addrepo " \
            "https://developer.download.nvidia.com/compute/cuda/repos/opensuse15/x86_64/cuda-opensuse15.repo;" \
            "echo a | zypper refresh;"
        execute_cmd(cuda_zypper_cmd)
        cuda_cmd = "zypper install -y cuda"
    elif 'Ubuntu' in distro:
        pre_cuda_setup_cmd = "apt-get install -y gcc-7 g++-7;" \
                             "update-alternatives --remove-all gcc;" \
                             "update-alternatives --remove-all g++;" \
                             "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 50;" \
                             "update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 50;" \
                             "update-alternatives --config gcc;" \
                             "update-alternatives --config g++;"
        execute_cmd(pre_cuda_setup_cmd)
        cuda_file = '../cuda_11.4.3_470.82.01_linux.run'
        cuda_cmd = "cd ${{IDIR}};" \
                   f"wget -N http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/{cuda_file};" \
                   f"sh {cuda_file} --silent --toolkit --samples;"
    else:
        cuda_cmd = f'echo "Failed; No CUDA install script for Linux Distribution: {distro}"'

    execute_cmd(cuda_cmd)


def install_borealis_env(python_version: str, user: str, group: str, no_cupy: bool = False):
    """
    Create virtual environment and install utilities needed for Borealis operation.

    :param  python_version: Python version to install venv as
    :type   python_version: str
    :param  user:           User to assign ownership permissions of the venv to
    :type   user:           str
    :param  group:          Group to assign ownership permissions of the venv to
    :type   group:          str
    :param  install_cupy:   Flag on whether to install cupy
    :type   install_cupy:   bool
    """
    print('### Creating Borealis virtual environment ###')

    execute_cmd(f"mkdir -p $BOREALISPATH/borealis_env{python_version}")
    execute_cmd(f"chown -R {user}:{group} $BOREALISPATH/borealis_env{python_version}")
    execute_cmd(f"sudo -u {user} python{python_version} -m venv $BOREALISPATH/borealis_env{python_version};")
    execute_cmd(f"sudo -u {user} $BOREALISPATH/borealis_env{python_version}/bin/python3 -m pip install wheel")
    pip_packages = ['zmq', 'numpy', 'scipy', 'protobuf==3.19.4', 'posix_ipc', 'structlog', 'graylog', 'rich',
                    'graypy', 'pydantic==1.10.11', # Temporary fix for pydantic code to work with python 3.11
                    'git+https://github.com/SuperDARN/pyDARNio.git@develop',
                    'git+https://github.com/SuperDARNCanada/backscatter.git#egg=backscatter']
    if not no_cupy:
        pip_packages.append('cupy')
    pkg_str = ''
    for pkg in pip_packages:
        pkg_str += f' {pkg}'
    execute_cmd(f"sudo -u {user} $BOREALISPATH/borealis_env{python_version}/bin/python3 -m pip install {pkg_str}")


def install_directories(user: str, group: str):
    """
    Install Borealis data and logging directories

    :param  user:   User to have user ownership permissions over the directories
    :type   user:   str
    :param  group:  Group to have group ownership permissions over the directories
    :type   group:  str
    """
    print('### Creating Borealis directories ###')
    mkdirs_cmd = "mkdir -p /data/borealis_logs;" \
                 "mkdir -p /data/borealis_data;" \
                 f"sudo -u {user} mkdir -p /home/{user}/logs;" \
                 f"chown {user}:{group} /data/borealis_*;"

    execute_cmd(mkdirs_cmd)


def install_hdw_dat():
    """
    Install hdw git repo
    """
    print('### Installing SuperDARN hdw repo ###')
    execute_cmd("git clone https://github.com/SuperDARN/hdw.git /usr/local/hdw")


def install_experiments(user: str, group: str):
    """
    Install Borealis experiment directory

    :param  user:   User to have user ownership permissions
    :type   user:   str
    :param  group:  Group to have group ownership permissions
    :type   group:  str
    """
    print('### Installing Borealis experiment files ###')
    install_experiments_cmd = "bash -c 'cd $BOREALISPATH';" \
                              f"sudo -u {user} git submodule update --init;" \
                              f"chown -R {user}:{group} src/borealis_experiments;"
    
    execute_cmd(install_experiments_cmd)


def main():
    parser = ap.ArgumentParser(usage=usage_msg(), description="Installation script for Borealis utils")
    parser.add_argument("--borealis-dir", help="Path to the Borealis installation directory", default="")
    parser.add_argument("--user", help="The username of the user who will run borealis. Default 'radar'",
                        default="radar")
    parser.add_argument("--group", help="The group name of the user who will run borealis. Default 'users'",
                        default="users")
    parser.add_argument("--python-version", help="The version of Python to use for the installation. Default 3.9",
                        default='3.9')
    parser.add_argument("--upgrade-to-v06", help="Is this to upgrade from Borealis v0.5 to v0.6?", action="store_true")
    parser.add_argument("--no-cuda", help="Do not install CUDA and cupy libraries.", action="store_true")
    parser.add_argument("radar", help="The three letter abbreviation for this radar. Example: sas")
    parser.add_argument("install_dir", help="Path to the installation directory")
    args = parser.parse_args()

    if os.geteuid() != 0:
        print('ERROR: You must run this script as root.')
        sys.exit(1)

    if not os.path.isdir(args.install_dir):
        print(f'ERROR: Install directory does not exist: {args.install_dir}')
        sys.exit(1)

    if args.borealis_dir == "":
        try:
            borealispath = os.environ['BOREALISPATH']
        except KeyError:
            print('ERROR: You must have an environment variable set for BOREALISPATH, or specify BOREALISPATH using --borealis-dir option.')
            sys.exit(1)
    else:
        os.environ['BOREALISPATH'] = args.borealis_dir

    distro = get_distribution()

    # Set env variables that will be read by subshells
    os.environ['IDIR'] = args.install_dir
    os.environ['CORES'] = str(mp.cpu_count())

    # Set up bash .profile RADAR_ID export
    radar_id = os.environ.get('RADAR_ID', None)
    if radar_id is None:
        execute_cmd(f'echo "export RADAR_ID={args.radar}" >> /home/{args.user}/.profile')
    elif radar_id != args.radar:
        raise ValueError(f'ERROR: RADAR_ID already specified as {radar_id}, cannot overwrite as {args.radar}')
    
    specify_python = False
    try:
        python_version = os.environ['PYTHON_VERSION']
        if python_version != args.python_version:
            raise ValueError(f'ERROR: PYTHON_VERSION already defined as {python_version}')
    except KeyError:
        specify_python = True

    if specify_python:
        print(f'### Specifying PYTHON_VERSION in /home/{args.user}/.profile ###')
        execute_cmd(f'echo "export PYTHON_VERSION={args.python_version}" >> /home/{args.user}/.profile')

    if args.upgrade_to_v06:     # Only need to update hdw repo and make new virtualenv for Borealis.
        print('### Upgrading to Borealis v0.6 configuration ###')
        install_hdw_dat()
        install_borealis_env(args.python_version, args.user, args.group, args.no_cuda)
        print('### REMINDER: Verify that your config.ini file conforms to the new format ###')

    else:   # Installing fresh, do it all!
        install_packages(distro)
        install_python(distro, args.python_version)
        pip_install_packages(args.user, args.python_version)
        install_protobuf()
        install_zmq()
        install_ntp()
        install_uhd(distro)
        if not args.no_cuda:
            install_cuda(distro)
        install_hdw_dat()
        install_borealis_env(args.python_version, args.user, args.group, args.no_cuda)
        install_directories(args.user, args.group)
        install_experiments(args.user, args.group)


if __name__ == '__main__':
    main()
