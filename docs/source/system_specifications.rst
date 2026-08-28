.. _parts:

=====================
System Specifications
=====================

-----------------------
Digital Radio Equipment
-----------------------

- **NOTE:** ALL cables are phase matched unless specified otherwise
- 17x Ettus USRP N200 (16 and 1 spare)

  - 17x Ettus LFTX daughterboards
  - 17x Ettus LFRX daughterboards
  - 17x Custom TXIO boards (for transmitter interfacing)

- 1x Ettus Octoclock-g (includes GPSDO)
- 2x Ettus Octoclock
- 51x ~8 1/4" SMA bulkhead Female to Male RG-316 for daughterboards (for 17x N200s, 3 cables each)
- 18x 48" SMA Male to Male RG-316 for PPS signals (2x from octoclock-g to octoclocks, 16x for N200s)
- 18x 48" SMA Male to Male RG-316 for 10MHz REF signals (2x from octoclock-g to octoclocks,
  16x for N200s)
- GPS Antenna (Male SMA connector)
- 17x sets of main array receive path circuitry (16 and 1 spare)
- 5x sets of interferometer array receive path circuitry (4 and 1 spare)
- Cabling and power for receive path circuitry

-------------------------
Borealis Control Computer
-------------------------

**Recommended Computer Hardware:** (Current control computer hardware as of September 2025)

- Case: Rosewill 4U Server Chassis Case
- Motherboard: GIGABYTE Z790 AORUS ELITE AX Motherboard
- CPU: Intel Core i9-12900K
- CPU Cooler: ASUS ROG Strix LC II 360
- GPU: ASUS GeForce RTX 4070 Ti
- Memory: 2x 16GB DDR5
- SSD Storage: 1x 2TB NVMe SSD
- HDD Storage: 2x 18TB 3.5" NAS HDD combined in RAID 1
- Power Supply: 1000W Gold rated
- Network Card: 10G Dual-Port SFP+ PCIe 3.0 x 8
- Serial Card: Modified serial card for PPS input

**Minimum Required Computer Hardware:**

- Motherboard: CPU compatible, with PCIe slots for both GPU and network card
- CPU: Intel Core i9-12900K
- CPU Cooler: Liquid cooler with 240mm radiator
- GPU: NVIDIA GeForce GTX 1080 Ti
- Memory: 24 GB DDR4
- SSD Storage: 256 GB SSD (OS partition)
- HDD Storage: 1 TB HDD (data partition)
- Power Supply: 1000W Gold rated
- Network Card: 10G PCIe 3.0 x 8

----------
Networking
----------

Below is the current network setup for the SuperDARN Canada radar sites:

- 1x FS S3900 48 Port 1GbE switch. Supports 10Mbps (for octoclock), 100Mbps, 1Gbps (for N200s), and
  10Gbps (for control computer) speeds
- 19x SSTP Cat6a 7ft cables (16x for the N200s, 3x for the octoclocks)
- 2x SFP+ 15ft Optical cables (for Borealis computer)
- Various Cat5e cables for peripheral network connected devices

Borealis requires at least Cat6a cables for connecting the N200s. Using less-shielded cables will
cause communication errors between the N200s. Cat5e and non-SSTP cables were tried during testing
and shown to **not** work - network cables need to be tested as not all cables seem to work
reliably.

This is just one network setup that works for Borealis, not the only one that works - other network
switches and cables can be used instead.

----------------
Rack and Cabling
----------------

- 1x 4-post 42U rack
- 2x 12-outlet rackmount power strip
- 2x APC AP7900B rackmount PDU
- 4x custom-made USRP N200 rackmount shelves
- 1x rackmount shelf for interferometer pre-amps
