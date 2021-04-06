========
Software
========

SuperDARN Canada uses OpenSUSE for an operating system, but any Linux system that can support the NVIDIA drivers for the graphics card will work.
The current latest version of OpenSuSe (15.1) is known to work. **Commands that require root privileges will have a `sudo` or `su` command ahead of them, or explicitly say 'as root', all others should be executed as the normal user (recommended name: radar) that will run Borealis**

#. Install the latest version of the NVIDIA drivers (see https://en.opensuse.org/SDB:NVIDIA_drivers). The driver must be able to support running the GPU selected and must also be compatible with the version of CUDA that supports the compute capability version of the GPU. Getting the OS to run stable with NVIDIA is the most important step, **so make sure you read this page carefully**.

#. Use the BIOS to find a stable over-clock for the CPU. Usually the recommended turbo frequency is a good place to start. This step is optional, but will help system performance when it comes to streaming high rates from the USRP. Do not adjust higher over-clock settings without doing research.

#. Use the BIOS to enable boot-on-power. The computer should come back online when power is restored after an outage. This setting is typically referred to as *Restore on AC/Power Loss*

#. Use cpupower to ungovern the CPU and run at the max frequency. This should be added to a script that occurs on reboot.

    - sudo cpupower frequency-set -g performance.

#. To verify that the CPU is running at maximum frequency:

    - cpupower frequency-info

#. Use ethtool to set the interface ring buffer size for both rx and tx. This should be added to a script that occurs on reboot for the interface used to connect to the USRPs. This is done to help prevent packet loss when the network traffic exceeds the capacity of the network adapter.

    - sudo ethtool -G eth0 tx 4096 rx 4096.

#. To see that this works as intended, and that it persists across reboots, you can execute the following, which will output the maximums and the current settings.

    - sudo ethtool -g eth0

#. Use the network manager or a line in the reboot script to change the MTU of the interface for the interface used to connect to the USRPs. A larger MTU will reduce the amount of network overhead. An MTU larger than 1500 bytes allows what is known as Jumbo frames, which can use up to 9000 bytes of payload.

    - sudo ip link set eth0 mtu 9000

#. To verify that the MTU was set correctly:

    - ip link show eth0

#. Use sysctl to adjust the kernel network buffer sizes. This should be added to a script that occurs on reboot for the interface used to connect to the USRPs. That's 50 million for `rmem_max` and 2.5 million for `wmwem_max`.

    - sudo sysctl -w net.core.rmem_max=50000000
    - sudo sysctl -w net.core.wmem_max=2500000

#. Verify that the kernel network buffer sizes are set:

    - cat /proc/sys/net/core/rmem_max
    - cat /proc/sys/net/core/wmem_max

#. The previous commands should all be executed on boot, as they are not persistent. One way to do this is via the root user's crontab. Here are some example entries for a computer with multiple ethernet interfaces:

    - @reboot /sbin/sysctl -w net.ipv6.conf.all.disable_ipv6=1
    - @reboot /sbin/sysctl -w net.ipv6.conf.default.disable_ipv6=1
    - @reboot /usr/sbin/ethtool -G eth1 tx 4096 rx 4096
    - @reboot /usr/sbin/ethtool -G eth2 tx 4096 rx 4096
    - @reboot /sbin/ip link set dev eth1 mtu 9000
    - @reboot /sbin/ip link set dev eth2 mtu 9000
    - @reboot /usr/bin/cpupower frequency-set -g performance
    - @reboot /sbin/sysctl -w net.core.rmem_max=50000000
    - @reboot /sbin/sysctl -w net.core.wmem_max=2500000

#. Install tuned. Use tuned-adm (as root) to set the system's performance to network-latency.

    - sudo zypper in tuned
    - su
    - systemctl enable tuned
    - systemctl start tuned
    - tuned-adm profile network-latency

#. To verify the system's new profile:

    - sudo tuned-adm profile_info

#. Add an environment variable called BOREALISPATH that points to the cloned git repository in .bashrc or .profile and re-source the file. For example:

    - export BOREALISPATH=/home/radar/borealis/
    - source .profile

#. Verify the BOREALISPATH environment variable exists:

    - env | grep BOREALISPATH

#. Clone the Borealis software to a directory **The following ensures that Borealis will be in the same directory that the `BOREALISPATH` env variable points to**.

    - sudo zypper in git
    - git clone https://github.com/SuperDARNCanada/borealis.git $BOREALISPATH

#. If you're building Borealis for a University of Saskatchewan radar, complete the following steps. If not, skip ahead to the next step. Create symlink `config.ini` in borealis directory and link to the site specific config file.

    - cd $BOREALISPATH
    - git submodule init && git submodule update
    - ln -svi $BOREALISPATH/borealis_config_files/[radarcode]_config.ini config.ini

#. If you're building Borealis for a non University of Saskatcheawn radar, use a Usask `config.ini` file (located `here <https://github.com/SuperDARNCanada/borealis_config_files>`_) as a template or the config file `documentation <https://borealis.readthedocs.io/en/latest/config_options.html>`_ to create your own file in the borealis directory.

#. The Borealis software has a script called `install_radar_deps.py` to help install dependencies. This script has to be run with root privileges. This script can be modified to add the package manager of a different distribution if it doesn't exist yet. Make sure that the version of CUDA is up to date and supports your card. This script makes an attempt to correctly install Boost and create symbolic links to the Boost libraries the UHD (USRP Hardware Driver) understands. If UHD does not configure correctly, an improper Boost installation or library naming convention is the likely reason. Note that you need python3 installed before you can run this script. If this script does not work for you, you can try the `install_radar_deps_opensuse.sh` shell script, which is known to work on OpenSuSe 15.1. The radar abbreviation should be the 3 letter radar code such as 'sas', 'rkn' or 'inv'.

    - cd $BOREALISPATH
    - chmod +x install_radar_deps.py
    - sudo ./install_radar_deps.py [radar abbreviation] $BOREALISPATH > install_log.txt 2>&1

#. Install pyDARNio for realtime data support as well as testing and data conversion support:

    - cd $HOME
    - git clone https://github.com/SuperDARN/pydarnio.git

#. Install the necessary software to enable realtime data:

    - cd /usr/local
    - sudo git clone https://github.com/vtsuperdarn/hdw.dat.git
    - mkdir $BOREALISPATH/borealisrt_env
    - virtualenv $BOREALISPATH/borealisrt_env
    - source $BOREALISPATH/borealisrt_env/bin/activate
    - pip install zmq
    - pip install git+git://github.com/SuperDARNCanada/backscatter.git#egg=backscatter
    - cd $HOME/pydarnio
    - git checkout develop
    - python setup.py install
    - deactivate

#. Install the necessary software to convert and test data:

    - cd $HOME
    - git clone https://github.com/SuperDARNCanada/borealis-data-utils.git
    - git clone https://github.com/SuperDARNCanada/data_flow.git
    - mkdir $HOME/pydarnio-env
    - virtualenv $HOME/pydarnio-env
    - source $HOME/pydarnio-env/bin/activate
    - cd $HOME/pydarnio
    - git checkout tags/v1.1.0 -b temp_branch  ### TODO: TEST THESE instructions for change from pydarn to pydarnio
    - python3 setup.py install
    - deactivate

#. Set up NTP. The `install_radar_deps_opensuse.sh` script already downloads and configures a version of `ntpd` that works with incoming PPS signals on the serial port DCD line. An example configuration of ntp is shown below for `/etc/ntp.conf`. These settings use `tick.usask.ca` as a time server, and PPS (via the `127.127.22.0` lines). It also sets up logging daily for all stats types.

    .. code-block::

        driftfile /var/log/ntp/ntp.drift

        statsdir /var/log/ntp/ntpstats/
        logfile /var/log/ntp/ntp_log
        logconfig =all
        statistics loopstats peerstats clockstats cryptostats protostats rawstats sysstats
        filegen loopstats file loopstats type day enable
        filegen peerstats file peerstats type day enable
        filegen clockstats file clockstats type day enable
        filegen cryptostats file cryptostats type day enable
        filegen protostats file protostats type day enable
        filegen rawstats file rawstats type day enable
        filegen sysstats file sysstats type day enable

        restrict -4 default kod notrap nomodify nopeer noquery limited
        restrict -6 default kod notrap nomodify nopeer noquery limited

        restrict 127.0.0.1
        restrict ::1

        restrict source notrap nomodify noquery

        server tick.usask.ca prefer
        server 127.127.22.0 minpoll 4 maxpoll 4
        fudge 127.127.22.0 time1 0.2 flag2 1 flag3 0 flag4 1

        keys /etc/ntp.keys
        trustedkey 1
        requestkey 1
        controlkey 1

#. As part of the realtime capabilities, the hdw.dat repo will be cloned to the computer(default will be /usr/local/hdw.dat). The hdw.dat files are also used for radar operation. Create a symbolic link for this radar in the $BOREALISPATH directory.

    - ln -svi /usr/local/hdw.dat/hdw.dat.[radarcode] $BOREALISPATH/hdw.dat.[radarcode]

#. Edit /etc/security/limits.conf (as root) to add the following line that allows UHD to set thread priority. UHD automatically tries to boost its thread scheduling priority, so it will fail if the user executing UHD doesn't have permission.

    - @users - rtprio 99

#. Assuming all dependencies are resolved, use `scons` to build the system. Use the script called `mode` to change the build environment to debug or release depending on what version of the system should be run. `SCONSFLAGS` variable can be added to `.bashrc/.profile` to hold any flags such as `-j` for parallel builds. For example, run the following:

    - cd $BOREALISPATH
    - `source mode [release|debug]`
    - If first time building, run `scons -c` to reset project state.
    - `scons` to build

#. Add the Python scheduling script, `start_radar.sh`, to the system boot scripts to allow the radar to follow the schedule. As an example on OpenSuSe for the `radar` user:

    - crontab -e
    - Add the line `@reboot /home/radar/borealis/start_radar.sh >> /home/radar/start_radar.log 2>&1`

#. Create necessary directories. Here is an example for a user named `radar` and the standard configuration in the 'config.ini' file:

    - sudo mkdir -p /data/borealis_logs
    - sudo mkdir -p /data/borealis_data
    - sudo chown radar:users /data/borealis_logs
    - sudo chown radar:users /data/borealis_data
    - mkdir $HOME/logs

#. Finally, add the GPS disciplined NTP lines to the root start up script.

    - /sbin/modprobe pps_ldisc && /usr/sbin/ldattach 18 /dev/ttyS0 && /usr/local/bin/ntpd

#. Verify that the PPS signal incoming on the DCD line of ttyS0 is properly routed and being received. You'll get two lines every second corresponding to an 'assert' and a 'clear' on the PPS line along with the time in seconds since the epoch.

    .. code-block::

        sudo ppstest /dev/pps0
        [sudo] password for root:
        trying PPS source "/dev/pps0"
        found PPS source "/dev/pps0"
        ok, found 1 source(s), now start fetching data...
        source 0 - assert 1585755247.999730143, sequence: 200 - clear  1585755247.199734241, sequence: 249187
        source 0 - assert 1585755247.999730143, sequence: 200 - clear  1585755248.199734605, sequence: 249188

#. For further reading on networking and tuning with the USRP devices, see https://files.ettus.com/manual/page_transport.html and https://kb.ettus.com/USRP_Host_Performance_Tuning_Tips_and_Tricks. Also see http://doc.ntp.org/current-stable/drivers/driver22.html for information about the PPS ntp clock discipline, and the man pages for:

    - tuned
    - cpupower
    - ethtool
    - ip
    - sysctl
    - modprobe
    - ldattach
