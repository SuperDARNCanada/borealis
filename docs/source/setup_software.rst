========
Software
========

SuperDARN Canada uses OpenSUSE for an operating system, but any Linux system that can support the NVIDIA drivers for the graphics card will work.
The current latest version of OpenSuSe (15.1) is known to work.

#. Install the latest version of the NVIDIA drivers (see https://en.opensuse.org/SDB:NVIDIA_drivers). The driver must be able to support running the GPU selected and must also be compatible with the version of CUDA that supports the compute capability version of the GPU. Getting the OS to run stable with NVIDIA is the most important step. You may need to add your linux user account to the 'video' group after installation.

#. Use the BIOS to find a stable over-clock for the CPU. Usually the recommended turbo frequency is a good place to start. This step is optional, but will help system performance when it comes to streaming high rates from the USRP. Do not adjust higher over-clock settings without doing research.

#. Use the BIOS to enable boot-on-power. The computer should come back online when power is restored after an outage. This setting is typically referred to as *Restore on AC/Power Loss*

#. Use cpupower to ungovern the CPU and run at the max frequency. This should be added to a script that occurs on reboot.

    - cpupower frequency-set -g performance.

#. To verify that the CPU is running at maximum frequency:

    - cpupower frequency-info

#. Use ethtool to set the interface ring buffer size for both rx and tx. This should be added to a script that occurs on reboot for the interface used to connect to the USRPs. This is done to help prevent packet loss when the network traffic exceeds the capacity of the network adapter.

    - ethtool -G eth0 tx 4096 rx 4096.

#. To see that this works as intended, and that it persists across reboots, you can execute the following, which will output the maximums and the current settings.

    - ethtool -g eth0

#. Use the network manager or a line in the reboot script to change the MTU of the interface for the interface used to connect to the USRPs. A larger MTU will reduce the amount of network overhead. An MTU larger than 1500 bytes allows what is known as Jumbo frames, which can use up to 9000 bytes of payload.

    - ip link set eth0 mtu 9000

#. To verify that the MTU was set correctly:

    - ip link show eth0

#. Use sysctl to adjust the kernel network buffer sizes. This should be added to a script that occurs on reboot for the interface used to connect to the USRPs.

    - sysctl -w net.core.rmem_max=50000000
    - sysctl -w net.core.wmem_max=2500000

#. Verify that the kernel network buffer sizes are set:

    - cat /proc/sys/net/core/rmem_max
    - cat /proc/sys/net/core/wmem_max

#. Install tuned. Use tuned-adm (as root) to set the system's performance to network-latency.

    - sudo zypper in tuned
    - su
    - systemctl enable tuned
    - systemctl start tuned
    - tuned-adm profile network-latency

#. To verify the system's new profile:

    - tuned-adm profile_info

#. Add an environment variable called BOREALISPATH that points to the cloned git repository in .bashrc or .profile and re-source the file. For example:

    - export BOREALISPATH=/home/radar/borealis/
    - source .profile

#. Clone the Borealis software to a directory.

    - git clone https://github.com/SuperDARNCanada/borealis.git
    - If Usask, git submodule init && git submodule update. Create symlink `config.ini` in borealis directory and link to the site specific config file.
    - cd ${BOREALISPATH} && ln -svi ${BOREALISPATH}/borealis_config_files/[radarcode]_config.ini config.ini
    - If not Usask, use a Usask `config.ini` file as a template or the config file documentation to create your own file in the borealis directory.

#. The Borealis software has a script called `install_radar_deps_opensuse.sh` to help install dependencies. This script has to be run by the root user. This script can be modified to use the package manager of a different distribution. Make sure that the version of CUDA is up to date and supports your card. This script makes an attempt to correctly install Boost and create symbolic links to the Boost libraries the UHD (USRP Hardware Driver) understands. If UHD does not configure correctly, an improper Boost installation or library naming convention is the likely reason.

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

    - ln -s /usr/local/hdw.dat/hdw.dat.[radarcode] $BOREALISPATH/hdw.dat.[radarcode]

#. Edit /etc/security/limits.conf to add the following line that allows UHD to set thread priority. UHD automatically tries to boost its thread scheduling priority, so it will fail if the user executing UHD doesn't have permission.

    - @users - rtprio 99

#. Assuming all dependencies are resolved, use `scons` to build the system. Use the script called `mode` to change the build environment to debug or release depending on what version of the system should be run. `SCONSFLAGS` variable can be added to `.bashrc/.profile` to hold any flags such as `-j` for parallel builds. For example, run the following:

    - `source mode [release|debug]`
    - If first time building, run `scons -c` to reset project state.
    - `scons` to build

#. Add the Python scheduling script, `start_radar.sh`, to the system boot scripts to allow the radar to follow the schedule.

#. Finally, add the GPS disciplined NTP lines to the root start up script.

    - /sbin/modprobe pps_ldisc && /usr/bin/ldattach 18 /dev/ttyS0 && /usr/local/bin/ntpd

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
