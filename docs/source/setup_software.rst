.. _software:

========
Software
========

SuperDARN Canada uses OpenSUSE Leap as the operating system for Boreals, but any Linux system that
can support the NVIDIA drivers for the graphics card should work. The latest version of OpenSUSE that
this installation has been tested on is **OpenSUSE Leap 15.6**.

**NOTE:** Commands that require root privileges will have a ``sudo`` or ``su`` command ahead of
them, or explicitly say 'as root', all others should be executed as the normal user that will run
Borealis (recommended name: radar).

#. Once the desired Linux operating system is installed on your computer, there are a few system
   settings that should be configured before installing any software.

   #. **Network Card Configuration:** Configure one of the 10G interfaces to the same subnet at the
      USRP devices - 192.168.10.x. The recommended 10G network card interface is as follows:

      - 10G interface 1: 192.168.1.x/24 (Interface for all other devices)
      - 10G interface 2: 192.168.10.x/24 (Interface for USRPs)

      Ensure IP routing is set up for each respective interface and subnet.

   #. **Hard Drive Partition:** The recommended hard drive partitions are shown below, given a single
      SSD and two HDDs. Adjust your partition sizes to match your hardware:

      - Split SSD into the following partitions:

         - Boot system partition (mounted to ``/boot/efi``)
         - Root partition (mounted to ``/``)
         - Home partition (mounted to ``/home``)
         - Swap partition

      - Combine both HDDs via RAID 1 and mount to single partition:

         - Data partition (mounted to ``/data``)

   #. **System Update**: Update all software packages via the package manager before proceeding with
      installation of other packages. This will limit issues when installing NVIDIA and CUDA drivers. ::

       sudo zypper update

#. Install the latest version of the NVIDIA drivers (see https://en.opensuse.org/SDB:NVIDIA_drivers).
   The driver must be able to support running the GPU
   selected and must also be compatible with the version of CUDA that supports the compute
   capability version of the GPU. Getting the OS to run stable with NVIDIA is the most important
   step, **so make sure you read this page carefully**.

   For our purposes the G06 drivers are required. To install the NVIDIA drivers on our OpenSUSE
   computers, first add the NVIDIA OpenSUSE repo: ::

    sudo zypper addrepo --refresh 'https://download.nvidia.com/opensuse/leap/$releasever' NVIDIA

   Then use ``zypper`` to install the NVIDIA drivers. For OpenSUSE Leap 15.5: ::

    sudo zypper install nvidia-video-G06 nvidia-gl-G06 nvidia-compute-utils-G06 nvidia-open

   For Leap 15.6 and Leap 16.0: ::

    sudo zypper install nvidia-open-driver-G06-signed-kmp-meta

   Reboot the computer. To verify that the drivers have been installed correctly, run
   ``sudo nvidia-smi`` - the output of this command should show the GPU installed on the computer.

   **NOTE:** It is possible to run Borealis on the CPU, that is, without using your graphics card for
   parallel computations. This will severely slow down the system, but may be useful in some cases. If
   this is desired, you can skip the step of installing NVIDIA drivers on your machine, and see
   the note when running ``install_radar_deps.py``.

#. Install the latest NVIDIA CUDA drivers (see
   https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). The radar software uses
   CUDA for accelerated data processing and is required for the best performance. Make sure the
   version of CUDA installed is appropriate for your GPU and works with your installed NVIDIA
   drivers.

   The following commands work to install CUDA on OpenSUSE Leap 15.6 and Leap 16.0: ::

    sudo zypper install -y kernel-<variant>-devel=<version>
    sudo usermod -a -G video <username>
    sudo zypper addrepo https://developer.download.nvidia.com/compute/cuda/repos/opensuse15/x86_64/cuda-opensuse15.repo
    sudo zypper install cuda-toolkit
    echo "export PATH=/usr/local/cuda/bin:${PATH}" >> ~/.profile
    source ~/.profile

   <variant> and <version> are retrieved from ``uname -r`` : ::

    $ uname -r
    <version>-<variant>

   Run ``nvcc --version`` to verify that CUDA is installed. If this command doesn't work, verify that
   ``/usr/local/cuda/bin`` is correctly added to the ``PATH`` environment variable within ``.profile``.

#. *Optional:* Use the BIOS to find a stable over-clock for the CPU. Usually the recommended turbo
   frequency is a good place to start. This step is optional, but will help system performance when
   it comes to streaming high rates from the USRP. Do not adjust higher over-clock settings without
   doing research. **NOTE: Overclocking is no longer suggested, as the increase performance comes
   at the cost of reliability**

#. Configure the computer to always remain powered on. This is done in two ways:

   #. Within BIOS configure the computer to always boot when connected to power. The computer should
      come back online when AC power is restored to the computer following a power outage. This setting
      is typically referred to as *Restore on AC/Power Loss* or *AC Power Recovery* or something
      similar.

   #. Ensure that the computer doesn't automatically hibernate or go to sleep. In OpenSUSE 16.0, the
      default setting when a user isn't actively logged in is to hibernate, after which the computer
      cannot be accessed via ``ssh``. To prevent this, create a ``sleep.conf`` file within 
      ``/etc/systemd/sleep.conf.d/`` (or wherever your ``sleep.conf.d/`` is located) containing the
      following lines: 

      ```ini
      [Sleep]
      AllowSuspend=no
      AllowHibernation=no
      AllowHybridSleep=no
      AllowSuspendThenHibernate=no
      ```

#. Configure the following computer settings to run each time the computer reboots. This can be done
   via root crontab, as these commands are not persistent. Example root crontab for multiple
   ethernet interfaces: ::

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

        sudo ethtool -G <10G_network_device> tx 4096 rx 4096

      To see that this works as intended, and that it persists across reboots, you can execute the
      following, which will output the maximums and the current settings. ::

        sudo ethtool -g <10G_network_device>

   #. Use ``ip`` to change the MTU for the interface used to connect to the USRPs. A larger MTU will
      reduce the amount of network overhead. An MTU larger than 1500 bytes allows what is known as
      Jumbo frames, which can use up to 9000 bytes of payload. **NOTE this also needs to be enabled
      on the network switch, and any other devices in the network chain. Setting this to 1500 may be
      the best option, make sure you test.** ::

        sudo ip link set <10G_network_device> mtu 9000

      To verify that the MTU was set correctly: ::

        ip link show <10G_network_device>

   #. Use ``cpupower`` to ungovern the CPU and run at the max frequency. This should be added to a
      script that occurs on reboot. ::

        sudo cpupower frequency-set -g performance

      To verify that the CPU is running at maximum frequency, ::

        cpupower frequency-info

   #. Use ``sysctl`` to adjust the kernel network buffer sizes. This should be added to a script
      that occurs on reboot for the interface used to connect to the USRPs. That's 50,000,000 for
      ``rmem_max`` and 2,500,000 for ``wmem_max``. ::

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

#. Create a .env file for the relevant Borealis environment variables. For example, in the home directory we define
   ``.borealis.env`` **(NOTE the extra '/' in BOREALISPATH)** ::

    BOREALISPATH=/home/radar/borealis/
    RADAR_ID=sas
    PYTHON_VERSION=3.13

#. Modify ``~/.profile`` to load the .env file with::

    set -a  # automatically export all variables
    source .borealis.env
    set +a

   Then, source the variables into your environment with ``source .profile``. Verify the ``BOREALISPATH`` environment
   variable exists::

    env | grep BOREALISPATH

#. Clone the Borealis software repository. Ensure that the ``BOREALISPATH`` environment variable
   points to the directory Borealis is cloned to. ::

    sudo zypper in git
    git clone https://github.com/SuperDARNCanada/borealis.git $BOREALISPATH

#. The Borealis software has a script called ``install_radar_deps.py`` to help install dependencies.
   The script is currently set up to install dependencies for OpenSUSE and Ubuntu. The following
   dependencies are installed within this script:

   #. The latest version of Python (version specified as a script argument)

   #. Packages required to run the radar, installed by the distribution package manager. Examples:
      gcc, UHD, C++ libraries, scons

   #. ZeroMQ and C++ bindings

   #. Creates the Borealis python environment and installs required python packages specified by
      ``pyproject.toml``

   #. Creates and clones additional directories needed for Borealis

   This script must be run with root privileges, as follows: ::

    cd $BOREALISPATH
    sudo -E python3 scripts/install_radar_deps.py [radar code] $BOREALISPATH --python-version=3.12 2>&1

   .. note::

      - You need ``python3`` installed before you can run this script.
      - The radar abbreviation should be the 3 letter radar code such as 'sas', 'rkn' or 'inv'.
      - This script can be modified to add the package manager of a different distribution if it
        doesn't exist yet.
      - If UHD does not configure correctly, an improper Boost installation or library naming
        convention is the likely reason. This script makes an attempt to correctly install Boost and
        create symbolic links to the Boost C++ libraries UHD (USRP Hardware Driver) understands.
      - If you do not have CUDA installed, pass the ``--no-cuda`` flag as an option.

#. Once all dependencies are resolved, use ``scons`` to compile the C++ code and build the system.
   ``SCONSFLAGS`` variable can be added to ``.profile`` to hold any flags such as ``-j`` for
   parallel builds. ::

    cd $BOREALISPATH
    scons -c          # If first time building, run to reset project state.
    scons release     # Can also run `scons debug`

#. Edit ``/etc/security/limits.conf`` (as root) to add the following lines.
   The first line allows UHD to set thread priority. UHD automatically tries to boost its thread scheduling priority, so it will fail
   if the user executing UHD doesn't have permission.
   The second line removes limits on the amount of page-locked memory and shared memory that a process can allocate
   for the ``radar`` user, which we use to prevent the ringbuffer from being swapped to disk. Switch ``radar`` with the user that
   will run borealis on your computer::

      radar - rtprio 99
      radar - memlock unlimited

#. *Optional:* Configure PPS signal input. A PPS signal is used to discipline NTP and improve timing
   to within microseconds - see :ref:`NTP Discipline with PPS<ntp-hardware>` for more info. This
   isn't neccessary to run the radar - using NTP without PPS is enough to properly run Borealis. To
   configure the PPS signal input follow these steps:

   #. Find out which tty device is physically connected to your PPS signal. It may not be ttyS0,
      especially if you have a PCIe expansion card. It may be ttyS1, ttyS2, ttyS3 or higher. To do
      this, search the system log for 'tty' (either ``dmesg`` or the ``syslog``). An example output
      with a PCIe expansion card is below. The output shows the first two (ttyS0 and ttyS1) built-in
      to the motherboard chipset are not accessible on this x299 PRO from MSI. The next two (ttyS4
      and ttyS5) are located on the XR17V35X chip which is located on the Rosewill card:

        .. code-block:: text

            [ 1.624103] serial8250: ttyS0 at I/O 0x3f8 (irq = 4, base_baud = 115200) is a 16550A
            [ 1.644875] serial8250: ttyS1 at I/O 0x2f8 (irq = 3, base_baud = 115200) is a 16550A
            [ 1.645850] 0000:b4:00.0: ttyS4 at MMIO 0xfbd00000 (irq = 37, base_baud = 7812500) is a XR17V35X
            [ 1.645964] 0000:b4:00.0: ttyS5 at MMIO 0xfbd00400 (irq = 37, base_baud = 7812500) is a XR17V35X

   #. Try attaching the ttySx line to a PPS line discipline using ``ldattach``: ::

        sudo /usr/sbin/ldattach PPS /dev/ttyS[0,1,2,3,etc]

   #. Verify that the PPS signal incoming on the DCD line of ttyS0 (or ttySx where x can be any
      digit 0,1,2,3...) is properly routed and being received. You'll get two lines every second
      corresponding to an 'assert' and a 'clear' on the PPS line along with the time in seconds
      since the epoch. If it's the incorrect one, you'll only see a timeout, so try attaching to
      a different ttySx input.

        .. code-block:: text

            sudo ppstest /dev/pps0
            trying PPS source "/dev/pps0"
            found PPS source "/dev/pps0"
            ok, found 1 source(s), now start fetching data...
            source 0 - assert 1585755247.999730143, sequence: 200 - clear  1585755247.199734241, sequence: 249187
            source 0 - assert 1585755247.999730143, sequence: 200 - clear  1585755248.199734605, sequence: 249188
            ...

   #. To determine which ``/dev/ttySx`` line is attached to the ``/dev/ppsx`` line tested above,
      look at the ``/sys/class/pps/ppsx/path`` file: ::

        radar@sasbore206:~> cat /sys/class/pps/pps0/path
        /dev/ttyS0

   #. Now create a small bash script that contains all commands needed to connect the GPS disciplined NTP line to
      the /dev/pps[X] line. This script will be used by chrony to configure the PPS line automatically when the 
      computer is turned on. Example script:

        .. code-block:: bash

             #!/bin/bash
             /sbin/modprobe pps_ldisc
             /usr/sbin/ldattach PPS /dev/ttyS[X]
             sleep 1

      Save this file as ``/usr/local/bin/pps-init.sh`` (for example) and ensure the file is executable. The 
      ``sleep 1`` command ensures that the correct `/dev/pps[X]` device is created before chrony tries to connect 
      to it.

#. Install and configure Network Time Protocol (NTP) via the built-in chrony package. chrony
   is an implementation of NTP native to OpenSUSE and Ubuntu that can be disciplined by an external
   PPS signal. **Note:** In past Borealis versions, NTP was installed directly from source binaries.

   #. An example ``/etc/chrony.conf`` configration file of chrony is shown below. These settings use
      ``time.usask.ca`` as an NTP server, which load balances between the ``tick.usask.ca`` and
      ``tock.usask.ca`` USask NTP servers. The optional PPS signal is configured on the ``refclock``
      line, where ``/dev/ppsx`` should be replaced with the correct PPS device determined in the
      previous section.

        .. code-block:: text

            server time.usask.ca iburst prefer
            refclock PPS /dev/ppsx refid PPS

            driftfile /var/lib/chrony/drift
            makestep 1.0 3
            rtcsync

            logdir /var/log/chrony
            log measurements statistics tracking

   #. *Optional:* To configure chrony with PPS, add the following lines to the ``chronyd.service``
      file: ::

        [Service]
        DeviceAllow=/dev/ttyS[X] rwm
        DeviceAllow=/dev/pps[X] rwm
        ExecStartPre=/usr/local/bin/pps-init.sh  # Modify if script name is different
        ProtectKernelModules=no  # Ensure that this setting isn't "yes" elsewhere in the file

   #. Start ``chronyd``: ::

        sudo systemctl enable chronyd.service
        sudo systemctl start chronyd.service

   #. To verify that chrony is working correctly, run ``chronyc sources -v``. The output should look
      similar to below: ::

        MS Name/IP address         Stratum Poll Reach LastRx Last sample
        ===============================================================================
        #* PPS                           0   4   377     9  +3022ns[+3134ns] +/- 1434ns
        ^- tick.usask.ca                 1  10   377   493   +921us[ +929us] +/-   16ms
        ^- tock.usask.ca                 1  10   377   554   -795us[ -788us] +/-   15ms
        ^- ntp1.yyz.ca.hojmark.net       2  10   377   228  -7684us[-7680us] +/-   62ms

   #. *Optional:* If PPS is not working with chrony, ``chronyc sources`` will look similar to below, 
      with an "x" next to PPS: ::

        MS Name/IP address         Stratum Poll Reach LastRx Last sample
        ===============================================================================
        #x PPS                           0   4     7     9   +465ms[ +465ms] +/-  376ns
        ^- tick.usask.ca                 1  10   377   493   +921us[ +929us] +/-   16ms
        ^* tock.usask.ca                 1  10   377   554   -795us[ -788us] +/-   15ms

      In this situation, the PPS signal is offset from the expected clock by 465 ms. To account for 
      this, modify the PPS line in ``/etc/chrony.conf`` as shown below. Adjust the offset to match
      what is shown in the ``chronyc sources`` table. ::

        refclock PPS /dev/ppsx refid PPS offset 0.465

      After restarting chrony, the sample should be in the microsecond range, instead of milliseconds.

#. If you're building Borealis for a non U of S radar, use one of the U of S ``[radar
   code]_config.ini`` files (located in ``borealis/config/[radar code]``) as a template, or follow
   the :ref:`config file documentation<config-options>` to create your own config file. Your config
   file should be placed in ``borealis/config/[radar code]/[radar code]_config.ini``

#. In ``[radar code]_config.ini``, there is an entry called "realtime_address". This defines the
   protocol, interface, and port that the realtime module uses for socket communication. This should
   be set to ``"realtime_address" : "tcp://<interface>:9696"``, where <interface> is a configured
   interface on your computer such as "127.0.0.1", "eth0", or "wlan0". This interface is selected
   from ``ip addr``, from which you should choose a device which is "UP".

   To verify that the realtime module is able to communicate with other modules, run the
   following command in a new terminal while Borealis is running. If all is well, the
   command should output that there is a device listening on the channel specified. ::

    ss --all | grep 9696

#. Configure the scheduler, and ensure Borealis runs. See :ref:`scheduling<scheduling>` for more
   information.

   #. Enable and start the ``atd`` system service. ``at`` is used to run the radar in specific modes
      following the radar schedule. ::

        sudo systemctl enable atd.service
        sudo systemctl start atd.service

   #. Ensure the site specific schedule files exists in the ``borealis_schedules`` directory (ex.
      ``sas.scd`` for Saskatoon).

   #. Run ``scripts/start_radar.sh`` to start the local radar scheduling server and attempt to start
      the radar. If the scheduler is working properly, the output of ``atq`` should show scheduled
      commands: ::

        radar@sasbore206:~> atq
        14748	Wed Sep 25 00:00:00 2024 a radar
        14749	Tue Oct  1 00:00:00 2024 a radar
        14750	Tue Oct  8 00:00:00 2024 a radar
        14751	Fri Oct 11 00:00:00 2024 a radar

   #. To check that the radar is operating, run ``screen -r borealis`` to view the live output of
      all Borealis processes.

#. Configure and install the automatic Borealis restart daemon, ``restart_borealis.service``. Follow
   the steps outlined here to install and start the system service: :ref:`Automated Restarts <automated-restarts>`.
   This daemon will automatically start the radar after five minutes, following the radar schedule.
   To verify that the daemon is working:

   - Check ``systemctl status restart_borealis.service`` that the system service is running
   - Check the logs at ``$HOME/logs/restart_borealis.log``

#. *Optional:* If your computer has enough RAM (greater than 16 GB), you can reduce how often swap
   is used. Borealis computers have occasionally utilized swap when RAM was still available, which
   greatly slows down Borealis resulting in very low sequence counts. To make this adjustment:

   #. Add ``vm.swappiness=10`` to ``/etc/sysctl.conf``. This changes the swap frequency from the
      default 60% to 10%.

   #. Run the following commands: ::

       sudo sysctl --load=/etc/sysctl.conf
       sudo swapoff -a
       sudo swapon -a

#. *Optional:* Configure the radar to operate in conjunction with a Uninterruptible Power Supply
   using ``apcupsd``. Follow the steps outlined here: :ref:`UPS & Power Outages <ups-power-outages>`.
   The scripts defined within ``borealis/scripts/apcupsd/`` can be used to turn the radar off/on
   when the UPS goes on battery.

#. Install necessary software to transfer, convert, and test data: ::

    cd $HOME
    git clone https://github.com/SuperDARNCanada/borealis-data-utils.git
    git clone https://github.com/SuperDARNCanada/data_flow.git
    python$PYTHON_VERSION -m venv $HOME/pydarnio-env
    source $HOME/pydarnio-env/bin/activate
    pip install pydarnio

   Follow the `data flow documentation <https://github.com/SuperDARNCanada/data_flow>`_ to properly
   setup and configure the data flow
