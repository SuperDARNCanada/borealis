SuperDARN Canada System Specifications
**************************************

=======================
Digital Radio Equipment
=======================

- Note that ALL cables are phase matched unless specified otherwise
- 17x Ettus USRP N200 (16 and 1 spare)
 - 17x Ettus LFTX daughterboards
 - 17x Ettus LFRX daughterboards
- 1x Ettus Octoclock-g (includes GPSDO)
- 2x Ettus Octoclock
- 51x ~8 1/4" SMA bulkhead Female to Male RG-316 for daughterboards
- 18x 48" SMA Male to Male RG-316 for PPS signals
- 18x 48" SMA Male to Male RG-316 for 10MHz REF signals
- 1x SMA Male to 0.1" pin header RG-316 for PPS signal input to motherboard
- GPS Antenna (Male SMA connector)
- 17x Custom TXIO Revision 5.0 board (for transmitter interfacing)
- 22x Mini-Circuits ZFL-500LN pre-amps (20 and 2 spare)
- 16x Mini-Circuites SLP-21.4 low pass filters
- 8x coax cables and adapters for to/from INTF (interferometer) pre-amps
- 32x coax cables for to/from main array filters and pre-amps inside transmitter
- 1x 15V, 0.5A power supply (INTF pre-amps)

================
Control Computer
================

- 1x GeForce GTX 2080 or better
- 2x 16GB DDR4
- 1x Monitor
- 1x Power supply, 1000W 80 Plus Gold or better
- 1x Intel Core i9 10 core or better
- 1x Cpu liquid cooling unit
- 1x CPU socket compatible motherboard with serial port header for PPS discipline
- 1x 256GB SSD 
- 1x 1TB HDD
- 1x Intel X550-T2 10Gb PCIe network card **NOTE**: Intel 82579LM controllers WILL NOT WORK


==========
Networking
==========

- 3x Netgear XS708E-200NES (North American model #) 10Gb switches (parent model name is XS708Ev2)
- 1x 5-port network switch that can handle 10Mbps and 100Mbps connection speeds (10BASE-T and 100BASE-T)
- 27x SSTP CAT 6a 7ft cables or better*
- 2x SSTP CAT 6a 15ft cables*

**Note** that the network cables need to be verified for the whole system
as not all cables seem to work reliably.

*Models tested and known to work include:*

- Cab-CAT6AS-05[GR|BK|GY|RE|WH]
- Cab-CAT6AS-15GR

*Models that were tested and do not work:*

- CAT 5e cables
- Non SSTP cables (not dual shielded)
- Cab-Cat7-6BL
- Cab-Cat7-6WH

================
Rack and Cabling
================

- 4x 8 outlet rackmount PDU
- 2x APC AP7900B rackmount PDU
- 1x 4 post 42U rack
- 4x custom-made USRP N200 rackmount shelves (or Ettus ones)
- 1x rackmount shelf for interferometer pre-amps

