Full System Setup Procedures
****************************

Here are the notes on SuperDARN Canada's Borealis setup procedures.

========
Hardware
========

-----
USRPs
-----

This guide assumes set up of a brand new, unopened unit.

1. Open the unit and install the LFTX/LFRX daughtercards. Use an SMA RF splitter pattern such that one end can split to a second splitter that connects TXA to RXA and RXB. The other splitter will attach to a scope.
2. USRPs have a default IP address of \192.168.10.2. Assign a computer network interface an address that can communicate in this subnet. Connect the USRP to the computer's network interface either directly or through one of the switches from the system specifications. Connect the USRP power supply.
3. Verify the board powers on and is discoverable. The USRP should be discoverable by pinging 192.168.10.2. Ettus' USRP UHD library supplies a tool called `uhd_usrp_probe`,`uhd_usrp_probe`should also be able to detect the device. See software setup for notes on installing UHD. The USRP may require a firmware upgrade.
4. Connect the free SMA output of the splitter to the scope. Connect the Octoclock PPS and 10MHz reference signals to the USRP. Make sure that the jumper on J510 is in the rightmost position connecting the front panel 10MHz as the system reference.
5. Use the UHD utilities `rx_samples_to_file`, `tx_bursts` and `txrx_loopback_to_file` to verify the USRP works. Use the scope to see the transmit signal. The RX samples will be a binary file that can be quickly read in a plotted with Numpy/Matplotlib. While testing, watch the front panel LEDs to see that they work as expected.
6. If the USRP is working correctly, the inner motherboard, fan, daughtercards and RF cables can all be removed from the unit. Carefully peel the product sticker. All removed components and the sticker can be stored in the anti-static bags that were supplied with the unit. The enclosure is ready for machining the additional holes.
7. Once the enclosures are machined, the electronics and components can all be reinstalled. Place the product sticker on the bottom left corner, closest to the front panel. Connect RXA to port RF1, connect RXB to port RF2, and connect TXA to the additional front panel hole that was added.
8. Install the LEDs and D-sub connector into the corresponding holes. The order of the LED install
patterns from left to right are the TX only indicator(RED), the idle indicator(YELLOW), the RX only indicator(GREEN) and the TR indicator(BLUE). Add labels to the LEDs on the front panel.
9. Install the TXIO board.

    - Begin by connecting eight female-female jumper cables to pins 1-4 and 6-9 of the D-sub connector on the inside of the USRP housing.
    - Next, connect pin 5 of J2 on the TXIO board to any digital ground pin available on the LFRX daughterboard connected to the N200. Do the same with pin 6 of J2.
    - Now we’ll connect the control signals. Connect the other end of the D-sub jumper cables to the pins of J2 as follows.

        - Connect pin 2 of J2 to pin 3 of the D-sub.
        - Connect pin 4 of J2 to pin 8 of the D-sub.
        - Connect pin 8 of J2 to pin 2 of the D-sub.
        - Connect pin 10 of J2 to pin 7 of the D-sub.
        - Connect pin 1 of J2 to pin 4 of the D-sub.
        - Connect pin 3 of J2 to pin 9 of the D-sub.
        - Connect pin 7 of J2 to pin 1 of the D-sub.
        - Connect pin 9 of J2 to pin 6 of the D-sub.

    - Once you finish with the J2 connections, we move on to the LFRX GPIO connections.

        - Connect pin 3 of J1 to an available digital ground pin on the LFRX daughterboard.
        - Connect pin 4 of J1 to the 6V power pin on the LFRX daughterboard.
        - Connect pin 5 of J1 to the io_rxa[1] pin on the LFRX daughterboard.
        - Connect pin 6 of J1 to the io_rxa[3] pin on the LFRX daughterboard.
        - Connect pin 7 of J1 to the io_rxa[5] pin on the LFRX daughterboard.
        - Connect pin 8 of J1 to the io_rxa[7] pin on the LFRX daughterboard.
        - Connect pin 9 of J1 to the io_rxa[11] pin on the LFRX daughterboard.
        - Connect pin 10 of J1 to the io_rxa[13] pin on the LFRX daughterboard.
        - Connect pin 11 of J1 to the io_rxa[9] pin on the LFRX daughterboard.
        - Connect pin 12 of J1 to an available digital ground pin on the LFRX daughterboard.

    - We connect the LEDs. Using female to female jumper cables, make the following connections:

        - Anode of red LED to J3 pin 1.
        - Cathode of red LED to J3 pin 2.
        - Anode of yellow LED to J3 pin 3.
        - Cathode of yellow LED to J3 pin 4.
        - Anode of green LED to J3 pin 5.
        - Cathode of green LED to J3 pin 6.
        - Anode of blue LED to J3 pin 7.
        - Cathode of blue LED to J3 pin 8.

    - Connect the outermost SMA-MFA cable to J7, proceeding inwards, connect the SMA-MFA cables to J6, J5, and J4 respectively.
    - Screw the TXIO board into place on the USRP housing.

10. Follow TXIO testing procedures from TXIO notes.
11. Install enclosure cover back in place.
12. Use UHD utility usrp_burn_mb_eeprom to assign a unique IP address for the unit. Label the unit with the device IP address.
13. The device should be configured and ready for use.

--------
Pre-amps
--------
TODO

----------
Rack Setup
----------

This guide will assume working top to bottom.

TODO

--------
Computer
--------

To be able to run Borealis at high data rates, a powerful CPU with many cores and a high number of
PCI lanes is needed. The team recommends an Intel i9 10 core CPU or better. Likewise a good NVIDIA
GPU is needed for fast data processing. The team recommends a GeForce 1080TI/2080 or better. Just
make sure the drivers are up to date on Linux for the model. A 10Gb or better network interface is
also required.

Once these components are selected, the supporting components such as motherboard, cooling and
hard drives can all be selected. Assemble the computer following the instructions that come with
the motherboard.

--------
Software
--------

SuperDARN Canada uses OpenSUSE for an operating system, but any Linux system that can support
the NVIDIA drivers for the graphics card will work.

1. Install the latest version of the NVIDIA drivers. The driver must be able to support running
the GPU selected and must also be compatible with the version of CUDA that supports the
compute capability version of the GPU. Getting the OS to run stable with NVIDIA is the most
important step.
2. Use the BIOS to find a stable over-clock for the CPU. Usually the recommended turbo frequency
is a good place to start. This step is optional, but will help system performance when it comes
to streaming high rates from the USRP. Do not adjust higher over-clock settings without doing
research.
3. Use cpupower to ungovern the CPU and run at the max frequency. This should be added to a script
that occurs on reboot.

    - cpupower frequency-set -g performance.

4. Use ethtool to set the interface ring size for both rx and tx. This should be added to a script
that occurs on reboot.

    - ethtool -G eth0 tx 4096 rx 4096.

5. Use sysctl to adjust the kernel network buffer sizes. This should be added to a script that
occurs on reboot.

    - sysctl -w net.core.rmem_max=50000000
    - sysctl -w net.core.wmem_max=2500000

6. Install tuned. Use tuned-adm to set the system's performance to network-latency.

    - tuned-adm profile network-latency

7. Clone the Borealis software to a directory.

    - git clone https://github.com/SuperDARNCanada/borealis.git

8. Add an environment variable called BOREALISPATH that holds to path to the cloned directory in
.bashrc or .profile and re-source the file.
9. The Borealis software has a script called install_radar_deps_opensuse.sh to help install
dependencies. This script can be modified to use the package manager of a different distribution.
Make sure that the version of CUDA is up to date and supports your card.
10. Assuming all dependencies are resolved, use scons to build the system. Use the script called
mode to change the build environment to debug or release depending on what version of the system
should be run.