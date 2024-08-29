.. _software:

========
Software
========

SuperDARN Canada uses OpenSUSE for an operating system, but any Linux system that can support the
NVIDIA drivers for the graphics card will work. The latest version of OpenSUSE that this
installation has been tested on is OpenSUSE Leap 15.5.

**NOTE:** Commands that require root privileges will have a ``sudo`` or ``su`` command ahead of
them, or explicitly say 'as root', all others should be executed as the normal user (recommended
name: radar) that will run Borealis.

**NOTE:** It is possible to run Borealis on the CPU, that is, without using your graphics card
for parallel computations. This will severely slow down the system, but may be useful in some cases.
If this is desired, you can skip the first step of installing NVIDIA drivers on your machine, and
see the note when running ``install_radar_deps.py``.

#. Install the latest version of the NVIDIA drivers (see
   https://en.opensuse.org/SDB:NVIDIA_drivers). The driver must be able to support running the GPU
   selected and must also be compatible with the version of CUDA that supports the compute
   capability version of the GPU. Getting the OS to run stable with NVIDIA is the most important
   step, **so make sure you read this page carefully**.

#. Install the latest NVIDIA CUDA drivers (see
   https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). The radar software uses
   CUDA for accelerated data processing and is required for the best performance. Make sure the
   version of CUDA installed is appropriate for your GPU and works with your installed NVIDIA drivers.

#.  **NOTE: Overclocking is no longer suggested, as reliability is more important than
    performance now**. Use the BIOS to find a stable over-clock for the CPU. Usually the recommended
    turbo frequency is a good place to start. This step is optional, but will help system performance
    when it comes to streaming high rates from the USRP. Do not adjust higher over-clock settings
    without doing research.

#. Use the BIOS to enable boot-on-power. The computer should come back online when power is restored
   after an outage. This setting is typically referred to as *Restore on AC/Power Loss*

#. Configure the following computer settings to run each time the computer reboots. This can be done via root
   crontab, as these commands are not persistent. Example root crontab for multiple ethernet interfaces: ::

    @reboot /sbin/sysctl -w net.ipv6.conf.all.disable_ipv6=1
    @reboot /sbin/sysctl -w net.ipv6.conf.default.disable_ipv6=1
    @reboot /usr/sbin/ethtool -G <10G_network_device_1> tx 4096 rx 4096
    @reboot /usr/sbin/ethtool -G <10G_network_device_2> tx 4096 rx 4096
    @reboot /sbin/ip link set dev <10G_network_device_1> mtu 9000
    @reboot /sbin/ip link set dev <10G_network_device_2> mtu 9000
    @reboot /usr/bin/cpupower frequency-set -g performance
    @reboot /sbin/sysctl -w net.core.rmem_max=50000000
    @reboot /sbin/sysctl -w net.core.wmem_max=2500000

   #. Use ``sysctl`` to disable IPv6. ::

        sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1
        sudo sysctl -w net.ipv6.conf.default.disable_ipv6=1

   #. Use ``ethtool`` to set the interface ring buffer size for both rx and tx. Make sure to use an
      ethernet device which is connected to the 10 GB card of the computer (not necessarily eth0). 
      This is done to help prevent packet loss when the network traffic exceeds the capacity of the
      network adapter. ::

        sudo ethtool -G <10G_network_device> tx 4096 rx 4096.

      To see that this works as intended, and that it persists across reboots, you can execute the
      following, which will output the maximums and the current settings. ::

        sudo ethtool -g <10G_network_device>

   #. Use ``ip`` to change the MTU for the interface used to connect to the USRPs. A larger MTU will 
      reduce the amount of network overhead. An MTU larger than 1500 bytes allows
      what is known as Jumbo frames, which can use up to 9000 bytes of payload. **NOTE this also needs
      to be enabled on the network switch, and any other devices in the network chain. Setting this
      to 1500 may be the best option, make sure you test.** ::

        sudo ip link set <10G_network_device> mtu 9000

      To verify that the MTU was set correctly: ::

        ip link show <10G_network_device>

   #. Use ``cpupower`` to ungovern the CPU and run at the max frequency. This should be added to a script
      that occurs on reboot. ::

        sudo cpupower frequency-set -g performance.

      To verify that the CPU is running at maximum frequency, ::

        cpupower frequency-info

   #. Use ``sysctl`` to adjust the kernel network buffer sizes. This should be added to a script that
      occurs on reboot for the interface used to connect to the USRPs. That's 50,000,000 for
      ``rmem_max`` and 2,500,000 for ``wmwem_max``. ::

        sudo sysctl -w net.core.rmem_max=50000000
        sudo sysctl -w net.core.wmem_max=2500000

      Verify that the kernel network buffer sizes are set: ::

        cat /proc/sys/net/core/rmem_max
        cat /proc/sys/net/core/wmem_max

#. Install ``tuned``. Use ``tuned-adm`` to set the system's performance to network-latency. ::

    sudo zypper in tuned
    sudo systemctl enable tuned
    sudo systemctl start tuned
    sudo tuned-adm profile network-latency

   To verify the system's new profile: ::

    sudo tuned-adm profile_info

#. Add an environment variable in ``.profile`` called ``BOREALISPATH`` that points to the cloned 
   Borealis git repository. For example **(NOTE the extra '/')**: ::

    export BOREALISPATH=/home/radar/borealis/
    source .profile

   Verify the ``BOREALISPATH`` environment variable exists: ::

    env | grep BOREALISPATH

#. Clone the Borealis software to a directory **The following ensures that Borealis will be in the
   same directory that the ``BOREALISPATH`` environment variable points to**. ::

    sudo zypper in git
    git clone https://github.com/SuperDARNCanada/borealis.git $BOREALISPATH

#. The Borealis software has a script called ``install_radar_deps.py`` to help install dependencies.
   This script has to be run with root privileges. This script can be modified to add the package
   manager of a different distribution if it doesn't exist yet. This script makes an attempt to
   correctly install Boost and create symbolic links to the Boost libraries the UHD (USRP Hardware
   Driver) understands. If UHD does not configure correctly, an improper Boost installation or library
   naming convention is the likely reason. Note that you need ``python3`` installed before you can run this
   script. The radar abbreviation should be the 3 letter radar code such as 'sas', 'rkn' or 'inv'.
   **NOTE:** If you do not have CUDA installed, pass the ``--no-cuda`` flag as an option. ::

    cd $BOREALISPATH
    sudo -E python3 scripts/install_radar_deps.py [radar code] $BOREALISPATH --python-version=3.11 2>&1 

#. If you're building Borealis for a non U of S radar, use one of the U of S
   ``[radar code]_config.ini`` files (located in ``borealis/config/[radar code]``) as a template, or follow the 
   :ref:`config file documentation<config-options>` to create your own config file. Your config file should
   be placed in ``borealis/config/[radar code]/[radar code]_config.ini``

#. In ``[radar code]_config.ini``, there is an entry called "realtime_address". This defines the protocol,
   interface, and port that the realtime module uses for socket communication. This should be set to
   ``"realtime_address" : "tcp://<interface>:9696"``, where <interface> is a configured interface on
   your computer such as "127.0.0.1", "eth0", or "wlan0". This interface is selected from ``ip addr``, 
   from which you should choose a device which is "UP".

   Verify that the realtime module is able to communicate with other modules. This can be done by
   running the following command in a new terminal while borealis is running. If all is well, the
   command should output that there is a device listening on the channel specified. ::

    ss --all | grep 9696

#. Edit ``/etc/security/limits.conf`` (as root) to add the following line that allows UHD to set
   thread priority. UHD automatically tries to boost its thread scheduling priority, so it will fail
   if the user executing UHD doesn't have permission. ::

    @users - rtprio 99

#. Once all dependencies are resolved, use ``scons`` to build the system. Use the script called
   ``mode`` to change the build environment to debug or release depending on what version of the
   system should be run. ``SCONSFLAGS`` variable can be added to ``.profile`` to hold any flags such
   as ``-j`` for parallel builds. ::

    cd $BOREALISPATH
    scons -c          # If first time building, run to reset project state.
    scons release     # Can also run `scons debug`

#. Configure PPS signal input. A PPS signal is used to discipline NTP and improve timing to within 
   microseconds - see :ref:`NTP Discipline with PPS<setup-hardware#ntp-discipline-with-pps>`_ for more info. 

   #. Find out which tty device is physically connected to your PPS signal. It may not be ttyS0,
      especially if you have a PCIe expansion card. It may be ttyS1, ttyS2, ttyS3 or higher. To do
      this, search the system log for 'tty' (either ``dmesg`` or the ``syslog``). An example output with a PCIe
      expansion card is below. The output shows the first two (ttyS0 and ttyS1) built-in to the
      motherboard chipset are not accessible on this x299 PRO from MSI. The next two (ttyS4 and ttyS5)
      are located on the XR17V35X chip which is located on the Rosewill card:

        .. code-block:: text

            [ 1.624103] serial8250: ttyS0 at I/O 0x3f8 (irq = 4, base_baud = 115200) is a 16550A
            [ 1.644875] serial8250: ttyS1 at I/O 0x2f8 (irq = 3, base_baud = 115200) is a 16550A
            [ 1.645850] 0000:b4:00.0: ttyS4 at MMIO 0xfbd00000 (irq = 37, base_baud = 7812500) is a XR17V35X
            [ 1.645964] 0000:b4:00.0: ttyS5 at MMIO 0xfbd00400 (irq = 37, base_baud = 7812500) is a XR17V35X

   #. Try attaching the ttySx line to a PPS line discipline using ``ldattach``: ::

        /usr/sbin/ldattach PPS /dev/ttyS[0,1,2,3,etc]

   #. Verify that the PPS signal incoming on the DCD line of ttyS0 (or ttySx where x can be any digit
      0,1,2,3...) is properly routed and being received. You'll get two lines every second
      corresponding to an 'assert' and a 'clear' on the PPS line along with the time in seconds since
      the epoch. If it's the incorrect one, you'll only see a timeout, and try a attaching to a different
      ttySx input.

        .. code-block:: text

            sudo ppstest /dev/pps0
            [sudo] password for root:
            trying PPS source "/dev/pps0"
            found PPS source "/dev/pps0"
            ok, found 1 source(s), now start fetching data...
            source 0 - assert 1585755247.999730143, sequence: 200 - clear  1585755247.199734241, sequence: 249187
            source 0 - assert 1585755247.999730143, sequence: 200 - clear  1585755248.199734605, sequence: 249188

   #. If you're having trouble finding out which ``/dev/ppsx`` device to use, try ``grep``ing the output of
      ``dmesg`` for pps to find out. Here's an example that shows how pps0 and pps1 are connected to ptp1 and ptp2, pps2
      is connected to ``/dev/ttyS0`` and pps3 is connected to ``/dev/ttyS5``.:

        .. code-block:: text

            [ 0.573439] pps_core: LinuxPPS API ver. 1 registered
            [ 0.573439] pps_core: Software ver. 5.3.6 - Copyright 2005-2007 Rodolfo Giometti <giometti@linux.it>
            [ 8.792473] pps pps0: new PPS source ptp1
            [ 9.040732] pps pps1: new PPS source ptp2
            [ 10.044514] pps_ldisc: PPS line discipline registered
            [ 10.045957] pps pps2: new PPS source serial0
            [ 10.045960] pps pps2: source "/dev/ttyS0" added
            [ 227.629896] pps pps3: new PPS source serial5
            [ 227.629899] pps pps3: source "/dev/ttyS5" added

#. Configure and start up NTP. The ``install_radar_deps.py`` script downloads and configures a version of
   ``ntpd`` that works with incoming PPS signals on the serial port DCD line. 
   
   #. An example configuration of ntp is shown below for ``/etc/ntp.conf``. These settings use ``tick.usask.ca``
      as a time server, with ``tock.usask.ca`` as a backup server, as well as PPS via the ``127.127.22.X`` 
      lines. **NOTE:** Replace the 'X' with the pps number that is connected to the incoming PPS signal determined 
      in the previous step (i.e. for pps0, PPS input is 127.127.22.1).

        .. code-block:: text

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
            server tock.usask.ca
            server 127.127.22.X minpoll 4 maxpoll 4
            fudge 127.127.22.X time1 0.2 flag2 1 flag3 0 flag4 1

            keys /etc/ntp.keys
            trustedkey 1
            requestkey 1
            controlkey 1

   #. Start ``ntpd``: ::

        sudo /usr/local/bin/ntpd

   #. To verify that ``ntpd`` is working correctly, run ``ntpq -p``: ::

        radar@rknmain207:~> ntpq --peers
             remote           refid      st t when poll reach   delay   offset  jitter
        ==============================================================================
        oPPS(1)          .PPS.            0 l    4   16  377    0.000   +2.662   1.317
        *tick.usask.ca   .GPS.            1 u   55   64  377   56.055   +0.545   2.186

      ``tick.usask.ca`` should have ``*`` in front of it, indicating that NTP is syncing
      to that server. ``PPS(X)`` should have ``o`` in front of it, indicating PPS is 
      being read successfully by NTP. 

      If PPS is not working correctly, follow the `NTP debug documentation <https://www.ntp.org/documentation/4.2.8-series/debug/>`_, and see
      `PPS Clock Discipline <http://www.fifi.org/doc/ntp-doc/html/driver22.htm>`_ for information about PPS.

#. Now add the GPS disciplined NTP lines to the root ``crontab`` on reboot using the tty you have your PPS
   connected to. This will start ``ntpd`` and attach the PPS signal on reboot. ::

    @reboot /sbin/modprobe pps_ldisc && /usr/sbin/ldattach PPS /dev/ttyS[X] && /usr/local/bin/ntpd

   For further reading on networking and tuning with the USRP devices, see
   `Transport Notes <https://files.ettus.com/manual/page_transport.html>`_ and
   `USRP Host Performance Tuning Tips and Tricks <https://kb.ettus.com/USRP_Host_Performance_Tuning_Tips_and_Tricks>`_.
   Also check out the man pages for ``tuned``, ``cpupower``, ``ethtool``, ``ip``, ``sysctl``,
   ``modprobe``, and ``ldattach``

#. Verify that the scheduler is working, and that the ``[radar code]].scd`` schedule file exists in the
   borealis_schedules directory.

#. Configure and install the automatic Borealis restart daemon, ``restart_borealis.service``. Follow the 
   steps outlined :ref:`here <starting-the-radar#automated-restarts>`_ to install and start the system service. This 
   daemon will automatically start the radar after five minutes, following the radar schedule. To 
   verify that the daemon is working:

   - Check ``systemctl status restart_borealis.service`` that the system service is running
   - Check the logs at ``$HOME/logs/restart_borealis.log``

#. Install necessary software to transfer, convert, and test data: ::

    cd $HOME
    git clone https://github.com/SuperDARNCanada/borealis-data-utils.git
    git clone https://github.com/SuperDARNCanada/data_flow.git
    python3.11 -m venv $HOME/pydarnio-env
    source $HOME/pydarnio-env/bin/activate
    pip install pydarn    # Installs pydarnio as well, as it is a dependency.

   Follow the `data flow documentation <https://github.com/SuperDARNCanada/data_flow>`_ to properly setup and
   configure the data flow
