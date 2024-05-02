.. _parts:

======================================
SuperDARN Canada System Specifications
======================================

-----------------------
Digital Radio Equipment
-----------------------

- **NOTE:** ALL cables are phase matched unless specified otherwise
- 17x Ettus USRP N200 (16 and 1 spare)

  - 17x Ettus LFTX daughterboards
  - 17x Ettus LFRX daughterboards

- 1x Ettus Octoclock-g (includes GPSDO)
- 2x Ettus Octoclock
- 51x ~8 1/4" SMA bulkhead Female to Male RG-316 for daughterboards (for 17x N200s, 3 cables each)
- 18x 48" SMA Male to Male RG-316 for PPS signals (2x from octoclock-g to octoclocks, 16x for N200s)
- 18x 48" SMA Male to Male RG-316 for 10MHz REF signals (2x from octoclock-g to octoclocks,
  16x for N200s)
- 1x SMA Male to 0.1" pin header RG-316 for PPS signal input to motherboard (homemade by cutting up
  a coaxial cable and soldering to a 0.1" two-position pin header)
- GPS Antenna (Male SMA connector)
- 17x Custom TXIO Revision 5.0 board (for transmitter interfacing)
- 22x Mini-Circuits ZFL-500LN pre-amps (20 and 2 spare)
- 16x Mini-Circuits SLP-21.4 low pass filters
- 8x coax cables and adapters for to/from INTF (interferometer) pre-amps
- 32x coax cables for to/from main array filters and pre-amps inside transmitter
- 1x 15V, 0.5A power supply (INTF pre-amps)

----------------
Control Computer
----------------

Current control computer hardware (1 January 2022):

- 1x CPU: Intel Core i9-12900K 3.2 GHz 16-Core Processor
- 1x CPU Cooler: Asus ROG RYUJIN II 360 71.6 CFM Liquid CPU Cooler
- 1x Motherboard: Asus ROG MAXIMUS Z690 EXTREME EATX LGA1700 Motherboard
- 1x Memory: G.Skill Trident Z5 32 GB (2 x 16 GB) DDR5-6000 CL36 Memory
- 2x Storage: Seagate FireCuda 530 2 TB M.2-2280 NVME Solid State Drive
- 1x Video Card: EVGA GeForce RTX 3080 Ti 12 GB FTW3 ULTRA GAMING LE iCX3 Video Card
- 1x Power Supply: SeaSonic PRIME Platinum 1300 W 80+ Platinum Certified Fully Modular ATX Power Supply
- 1x PCIe Serial Card: Rosewill RC-301EU (modified to replace dsub with sma conn)

**NOTE:** Always update the BIOS before installing the computer in the field.

**NOTE:** The motherboard has a 10G and 1G port, an additional network card is not required.
Optionally, Fiber optic cards can be used as a link to network switches that support them.

**NOTE:** XMP must be enabled in BIOS to utilize the full 6000 MHz RAM speed. RAM must also be socketed in
the optimal configuration DIMM_A2 and DIMM_B2 for two sticks.

**NOTE:** The M.2 drives should be slotted in sockets M.2_2 and M.2_3 as M.2_1 shares PCIe Gen5 lanes
with PCIex16(G5)_1 which the GPU is slotted into. The M.2 drives are setup in RAID1 (mirroring) including
the operating system. The bootloader on the second drive must be named differently to operate. Should drive
failure occur the second bootloader must be pointed to in the BIOS, a new drive can then be installed.

**NOTE:** It is critical during parts selection and installation that the PCIe lanes are dedicated and not
split between components. Seamless operation requires the maximization of bandwidth. For the Z690 motherboard
this means the GPU is slotted in PCIex16(G5)_1 and the Serial Card is slotted in PCIex1(G3), PCIex18(G5)_2
is not to be filled as it share bandwidth with PCIex16(G5)_1.

**NOTE:** A 1000W+ platinum or greater certified power supply is recommended.

**NOTE:** A 3x120mm (360mm radiator) All-in-one (AIO) liquid cooler heatsink is required. The i9 series CPUs
generate considerable heat by default, but the Borealis computer setup will pin each core to its maximum
output which will generate even more heat.

**NOTE:** This computer does not require overclocking for seamless operation which will allow for more
reliable operation than previous Borealis computer builds.

Minimum requirements:

- 1x GeForce GTX 2080 Ti with 11GB of GRAM or better
- 2x 16GB DDR4 (32GB total) 3200 MHz or faster
- 1x Power supply, 1000W 80 Plus Gold or better
- 1x Intel Core i9 10 core or better
- 1x Cpu liquid cooling unit (240mm or 360mm)
- 1x CPU compatible motherboard with serial port header or an extra unshared bandwidth PCIe slot for a serial card
- 1x 256GB SSD (operating system partition)
- 1x 1TB HDD (data partition)
- 1x Intel X550-T2 10Gb PCIe network card (if the motherboard does not have 10G networking) OR
  optionally, a Fiber optic NIC such as the Mellanox MCX4121A-ACAT, or the Intel X710-BM2

**NOTE:** Intel 82579LM controllers WILL NOT WORK

**NOTE:** A BIOS flash is required to use a RTX3000+ series NVidia GPU.

----------
Networking
----------

- 3x Netgear XS708E-200NES (North American model #) 10Gb switches (parent model name is XS708Ev2)
  (NOTE: These network switches are discontinued as of 2021. See `this page
  <https://community.netgear.com/t5/Plus-and-Smart-Switches-Forum/XS708T-and-XS716T-discontinued/m-p/2137635>`_)
- A potential replacement with a fiber optic link (SFP+) is the FS S3900 48 Port 1GbE switch.
- 1x 5-port network switch that can handle 10Mbps and 100Mbps connection speeds (10BASE-T and 100BASE-T)
  OR if using the FS S3900 48 port, it supports 10Mbps for the octoclocks.
- 27x SSTP CAT 6a 7ft cables or better* (16x for the N200s, 2x for daisy-chaining the switches, 3x
  for the octoclocks, 1x for connecting to the 1GbE network switch, and 5x spares).
- 2x SSTP CAT 6a 15ft cables* (for connecting to the Borealis computer, and one spare)
- Optional for Fiber: DAC/AOC cables such as the FS SFPP-AO05

**NOTE:** network cables need to be verified for the whole system as not all cables seem to work
reliably.

*Models tested and known to work include:*

- Cab-CAT6AS-05[GR|BK|GY|RE|WH]
- Cab-CAT6AS-15GR

*Models that were tested and do not work:*

- CAT 5e cables
- Non SSTP cables (not dual shielded)
- Cab-Cat7-6BL
- Cab-Cat7-6WH

----------------
Rack and Cabling
----------------

- 4x 8 outlet rackmount PDU
- 2x APC AP7900B rackmount PDU (minimum, a third would be useful)
- 1x 4 post 42U rack
- 4x custom-made USRP N200 rackmount shelves (or Ettus ones)
- 1x rackmount shelf for interferometer pre-amps
