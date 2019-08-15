========
Hardware
========

-----
USRPs
-----

This guide assumes set up of a brand new, unopened unit.

Initial Test of the Unit
------------------------

*Install Daughterboards*

1. Open the unit and install the LFTX and LFRX daughtercards using hardware provided. The main USRP PCB is clearly marked with where to connect TX and RX daughterboards. ???
2. Connect the output of TXA using an SMA cable to the custom-added SMA connection point on the front of the USRP. Connect the output of RXA to RF1 and RXB to RF2 on the front of the USRP. 

*Connect to the USRP*

3. USRPs have a default IP address of 192.168.10.2. Assign a computer network interface an address that can communicate in this subnet. Connect the USRP to the computer's network interface either directly or through one of the switches from the system specifications. Connect the USRP power supply.
4. Verify the board powers on and is discoverable. The USRP should be discoverable by pinging 192.168.10.2. Ettus' USRP UHD library supplies a tool called `uhd_usrp_probe`. `uhd_usrp_probe` should also be able to detect the device. See software setup for notes on installing UHD. The USRP may require a firmware upgrade.
5. Connect the free SMA output of the splitter to the scope. Connect the Octoclock PPS and 10MHz reference signals to the USRP. Make sure that the jumper on J510 is in the rightmost position connecting the front panel 10MHz as the system reference.

*Test the USRP*

6. Use an SMA RF splitter pattern such that one end can split to a second splitter that connects TXA to RXA and RXB. The other splitter will attach to a scope.
7. Use the UHD utilities `rx_samples_to_file`, `tx_bursts` and `txrx_loopback_to_file` to verify the USRP works. Use the scope to see the transmit signal. The RX samples will be a binary file that can be quickly read in a plotted with Numpy/Matplotlib. While testing, watch the front panel LEDs to see that they work as expected.

*Disassembly for Enclosure Modifications*

8. If the USRP is working correctly, the inner motherboard, fan, daughtercards and RF cables can all be removed from the unit. Carefully peel the product sticker and store with the motherboard. All removed components and the sticker can be stored in the anti-static bags that were supplied with the unit. The enclosure is ready for machining the additional holes.

Custom Enclosure Modifications
------------------------------

*TODO*


Installing the Custom-Made TXIO Board
-------------------------------------

1. Once the enclosures are machined, the electronics and components can all be reinstalled. Place the product sticker on the bottom left corner, closest to the front panel. Connect RXA to port RF1, connect RXB to port RF2, and connect TXA to the additional front panel hole that was added.
2. Install the LEDs and D-sub connector into the corresponding holes. The order of the LED install patterns from left to right are the TX only indicator(RED), the idle indicator(YELLOW),the RX only indicator(GREEN) and the TR indicator(BLUE). Add labels to the LEDs on the front panel.
3. Install the TXIO board.

    - Begin by connecting eight female-female jumper cables to pins 1-4 and 6-9 of the D-sub connector on the inside of the USRP housing.
    - Next, connect pin 5 of J2 on the TXIO board to any digital ground pin available on the LFRX daughterboard connected to the N200. Do the same with pin 6 of J2.
    - Now connect the control signals. Connect the other end of the D-sub jumper cables to the pins of J2 as follows.

        - Connect pin 2 of J2 to pin 3 of the D-sub.
        - Connect pin 4 of J2 to pin 8 of the D-sub.
        - Connect pin 8 of J2 to pin 2 of the D-sub.
        - Connect pin 10 of J2 to pin 7 of the D-sub.
        - Connect pin 1 of J2 to pin 4 of the D-sub.
        - Connect pin 3 of J2 to pin 9 of the D-sub.
        - Connect pin 7 of J2 to pin 1 of the D-sub.
        - Connect pin 9 of J2 to pin 6 of the D-sub.

    - Once you finish with the J2 connections, move on to the LFRX GPIO connections.

        - Connect pin 3 of J1 to an available digital ground pin on the LFRX daughterboard.
        - Connect pin 4 of J1 to the 6V power pin on the LFRX daughterboard.
        - Connect pin 5 of J1 to the io_rxa[1] pin on the LFRX daughterboard.
        - Connect pin 6 of J1 to the io_rxa[3] pin on the LFRX daughterboard.
        - Connect pin 7 of J1 to the io_rxa[5] pin on the LFRX daughterboard.
        - Connect pin 8 of J1 to the io_rxa[7] pin on the LFRX daughterboard.
        - Connect pin 9 of J1 to the io_rxa[9] pin on the LFRX daughterboard.
        - Connect pin 10 of J1 to the io_rxa[11] pin on the LFRX daughterboard.
        - Connect pin 11 of J1 to the io_rxa[13] pin on the LFRX daughterboard.
        - Connect pin 12 of J1 to an available digital ground pin on the LFRX daughterboard.

    - Connect the LEDs. Using female to female jumper cables, make the following connections:

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

4. Follow the testing procedure below to run a simple test of the TXIO outputs.

**TXIO OUTPUT TESTS**

- Connect a needle probe to channel one of your oscilloscope and set it to trigger on the rising edge of channel one.

- Run test_txio_gpio.py located in borealis/testing/n200_gpio_test. Usage is as follows:

    `python3 test_txio_gpio.py <N200_ip_address>`

- When prompted to enter the pins corresponding to the TXIO signals, press enter to accept the default pin settings. This will begin the tests. Pressing CTRL+C and entering "y" will tell the program to run the next test.

- Insert the needle probe into the SMA output corresponding to RXO. The scope signal should be the inverse of the pattern flashed by the GREEN front LED. Then, proceed to the next test (CTRL+C, then enter "y").

- Insert the needle probe into the SMA output corresponding to TXO. The scope signal should be the inverse of the pattern flashed by the RED and BLUE front LEDs. Then, proceed to the next test (CTRL+C, then enter "y").

- Insert the needle probe into the SMA output corresponding to TR. The scope signal should be the inverse of the pattern flashed by the BLUE and GREEN front LEDs. Then, proceed to the next test (CTRL+C, then enter "y").

    - Insert the needle probe into the hole corresponding to pin 7 of the D-Sub connector (TR+). The scope signal should follow the pattern flashed by the BLUE and GREEN front LEDs.

    - Insert the needle probe into the hole corresponding to pin 2 of the D-Sub connector (TR-). The scope signal should be the inverse of the pattern flashed by the BLUE and GREEN front LEDs.

- Insert the needle probe into SMA output corresponding to IDLE. The scope signal should be the inverse of the pattern flashed by the YELLOW front LED. Then, proceed to the next test (CTRL+C, then enter "y").

- Insert the needle probe into the hole corresponding to pin 8 of the D-Sub. The scope signal should follow the sequence of numbers being printed to your terminal (high when the number is non-zero, low when the number is zero).

    - Insert the needle probe into the hole corresponding to pin 3 of the D-Sub. The scope signal should be the inverse of the sequence of numbers being printed to your terminal. Then, proceed to the next test (CTRL+C, then enter "y").

- To properly perform the loopback tests of the differential signals, connect the D-Sub pins to each other in the following configuration:

    - Pin 6 to pin 7
    - Pin 1 to pin 2
    - Pin 8 to pin 9
    - Pin 3 to pin 4

- Once connected ensure that during the TR, AGC loopback test, the printed number is non zero when the terminal indicates the output pin is low, and vice versa. Then, proceed to the next test (CTRL+C, then enter "y").

- Ensure that during the TM, LP loopback test, the printed number is non zero when the terminal indicates the output pin is low, and vice versa. Press CTRL+C, then enter "y" to end the tests.

- This concludes the tests! If any of these signal output tests failed, additional troubleshooting is needed. To check the entire logic path of each signal, follow the testing procedures found in the TXIO notes document.

5. Install enclosure cover lid back in place.

Configuring the Unit for Borealis
---------------------------------

1. Use UHD utility usrp_burn_mb_eeprom to assign a unique IP address for the unit. Label the unit with the device IP address.
2. The device should be configured and ready for use.


--------
Pre-amps
--------

For easy debugging, pre-amps are recommended to be installed inside existing SuperDARN transmitters where possible for SuperDARN main array channels. SuperDARN transmitters typically have a 15V supply and the low-noise amplifiers selected for pre-amplification (Mini-Circuits ZFL-500LN) operate at 15V, with max 60mA draw. The cable from the LPTR (low power transmit/receive) switch to the bulkhead on the transmitter can be replaced with a couple of cables to and from a filter and pre-amp. 

Note that existing channel filters (typically custom 8-20MHz filters) should be placed ahead of the pre-amps in line to avoid amplifying noise. 

It is also recommended to install all channels the same for all main array channels to avoid varying electrical path lengths in the array which will affect beamformed data.

Interferometer channels will need to be routed to a separate plate and supplied with 15V by a separate supply. 

----------
Rack Setup
----------

Below is a recommended configuration in comparison to a common SuperDARN system:

.. figure:: USRP-rack-rev3.png
   :scale: 75 %
   :alt: Block diagram of RX DSP software
   :align: center

Here is an actual rack configuration as installed by SuperDARN Canada at the Saskatoon (SAS) SuperDARN site. Note that space has been allowed between the rackmount items to allow for cable routing. There is a lot of cabling involved at the front of the devices.

.. figure:: sas-borealis-rack.jpg
   :scale: 75 %
   :alt: Block diagram of RX DSP software
   :align: center

The items installed in the rack at the Saskatoon site are listed below in order from bottom to top in the rack:

- APC PDU (AP7900B)
- 15V Acopian power supply
- APC Smart UPS
- Custom-made logic signal testing box using Saleae logic analyzer (for test purposes only)
- TrippLite power bar
- Netgear XS708E 10Gb switch
- USRP rackmount shelf (Ettus manufactured) with 4 x N200s
- Ettus Octoclock
- USRP rackmount shelf (Ettus manufactured) with 4 x N200s
- Rackmount shelf with 4 x low-noise amplifiers for the interferometer array channels, and a terminal strip for power (supplied by 15V Acopian)
- Ettus Octoclock-G (with GPSDO)
- Netgear XS708E 10Gb switch
- APC PDU (AP7900B)
- USRP rackmount shelf (Ettus manufactured) with 4 x N200s
- Ettus Octoclock
- USRP rackmount shelf (Ettus manufactured) with 4 x N200s
- Netgear XS708E 10Gb switch
- APC PDU (AP7900B)

You can also see the Borealis computer at this site is not in a rackmount case, instead it is shown to the right of the rack. 


-----------------------
Computer and Networking
-----------------------

To be able to run Borealis at high data rates, a powerful CPU with many cores and a high number of PCI lanes is needed. The team recommends an Intel i9 10 core CPU or better. Likewise a good NVIDIA GPU is needed for fast data processing. The team recommends a GeForce 1080TI/2080 or better. Just make sure the drivers are up to date on Linux for the model. A 10Gb(or multiple 1Gb interfaces) or better network interface is also required.

Not all networking equipment works well together or with USRP equipment. Some prototyping with different models may be required.

Once these components are selected, the supporting components such as motherboard, cooling and hard drives can all be selected. Assemble the computer following the instructions that come with the motherboard.