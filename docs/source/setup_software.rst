========
Software
========

SuperDARN Canada uses OpenSUSE for an operating system, but any Linux system that can support the NVIDIA drivers for the graphics card will work.

#. Install the latest version of the NVIDIA drivers (see https://en.opensuse.org/SDB:NVIDIA_drivers). The driver must be able to support running the GPU selected and must also be compatible with the version of CUDA that supports the compute capability version of the GPU. Getting the OS to run stable with NVIDIA is the most important step. You may need to add your linux user account to the 'video' group after installation.

#. Use the BIOS to find a stable over-clock for the CPU. Usually the recommended turbo frequency is a good place to start. This step is optional, but will help system performance when it comes to streaming high rates from the USRP. Do not adjust higher over-clock settings without doing research.

#. Use the BIOS to enable boot-on-power. The computer should come back online when power is restored after an outage.

#. Use cpupower to ungovern the CPU and run at the max frequency. This should be added to a script that occurs on reboot.

    - cpupower frequency-set -g performance.

#. Use ethtool to set the interface ring size for both rx and tx. This should be added to a script that occurs on reboot for the interface used to connect to the USRPs.

    - ethtool -G eth0 tx 4096 rx 4096.

#. Use the network manager or a line in the reboot script to change the MTU of the interface for the interface used to connect to the USRPs.

    - ip link set eth0 mtu 9000

#. Use sysctl to adjust the kernel network buffer sizes. This should be added to a script that occurs on reboot for the interface used to connect to the USRPs.

    - sysctl -w net.core.rmem_max=50000000
    - sysctl -w net.core.wmem_max=2500000

#. Install tuned. Use tuned-adm (as root) to set the system's performance to network-latency.

    - sudo zypper in tuned
    - su
    - systemctl enable tuned
    - systemctl start tuned
    - tuned-adm profile network-latency

#. Add an environment variable called BOREALISPATH that points to the cloned git repository in .bashrc or .profile and re-source the file. For example:

    - export BOREALISPATH=/home/radar/borealis/
    - source .profile

#. Clone the Borealis software to a directory.

    - git clone https://github.com/SuperDARNCanada/borealis.git
    - If Usask, git submodule init && git submodule update. Create symlink `config.ini` in borealis directory and link to the site specific config file.
    - cd ${BOREALISPATH} && ln -svi ${BOREALISPATH}/borealis_config_files/[radarcode]_config.ini config.ini
    - If not Usask, use a Usask `config.ini` file as a template or the config file documentation to create your own file in the borealis directory.

#. The Borealis software has a script called install_radar_deps_opensuse.sh to help install dependencies. This script has to be run by the root user. This script can be modified to use the package manager of a different distribution. Make sure that the version of CUDA is up to date and supports your card. This script makes an attempt to correctly install Boost and create symbolic links to the Boost libraries the UHD understands. If UHD does not configure correctly, an improper Boost installation or library naming convention is the likely reason.

#. As part of the realtime capabilities, the hdw.dat repo will be cloned to the computer(default will be /usr/local/hdw.dat). The hdw.dat files are also used for radar operation. Create a symbolic link for this radar in the $BOREALISPATH directory.

    - ln -s /usr/local/hdw.dat/hdw.dat.[radarcode] $BOREALISPATH/hdw.dat.[radarcode]

#. Edit /etc/security/limits.conf to add the following line that allows UHD to set thread priority.

    - @users - rtprio 99

#. Assuming all dependencies are resolved, use scons to build the system. Use the script called `mode` to change the build environment to debug or release depending on what version of the system should be run. `SCONSFLAGS` variable can be added to .bashrc/.profile to hold any flags such as -j for parallel builds. For example, run the following:

    - source mode [release|debug]
    - If first time building, run `scons -c` to reset project state.
    - `scons` to build

#. Add the Python scheduling script, `start_radar.sh`, to the system boot scripts to allow the radar to follow the schedule.

#. Finally, add the GPS disciplined NTP lines to the root start up script.

   - /usr/sbin/ldattach 18 /dev/ttyS0
   - /usr/local/bin/ntpd

