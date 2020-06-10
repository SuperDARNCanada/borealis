Common Failure modes
====================

Certain failures of the Borealis system can be generally attributed to a short list of common issues.

N200 Power loss
---------------
If the power to any N200 device that Borealis is currently using is disrupted for a brief time,
then the symptoms are typically:

    - Driver message: "Timed out! RuntimeError fifo ctrl timed out looking for acks"
    - The N200 that lost power will have all front panel aftermarket LEDs ON
    - All other N200s will have the green RX LED ON.
    - Radar stops

Restart the radar by:

    - Ensuring the power is securely connected to all N200s
    - `/borealis/stop_radar.sh`
    - `/borealis/start_radar.sh`

N200 10MHz reference loss
-------------------------
If the 10MHz reference signal to any N200 device that Borealis is currently using is disrupted for
an extended time (beyond a few seconds) then the symptoms are:

    - Continual 'lates' from the driver ('L' printed out continuously)
    - `REF locked` front panel LED will be off for the N200 that lost 10MHz reference
    - Upon reconnection of the 10MHz signal, the lates continue
    - Radar continues

Restart the radar by:

    - Ensuring the 10MHz reference is connected to all N200s
    - `/borealis/stop_radar.sh`
    - `/borealis/start_radar.sh`

N200 PPS reference loss
-----------------------
If the Pulse Per Second (PPS) reference signal to any N200 device that Borealis is currently using
is disrupted for an extended time (beyond a few seconds) then the symptoms are:

    - None

N200 Ethernet loss
------------------
If the ethernet connection to any N200 device that Borealis is currently using is disrupted for
a brief time, the symptoms are typically:

    - Borealis software hangs
    - After some time, the aftermarket front panel LEDS turn yellow, indicating an IDLE situation
    - Radar stops

Restart the radar by:

    - Reconnecting the Ethernet
    - `/borealis/stop_radar.sh`
    - `/borealis/start_radar.sh`


Borealis Startup with N200 PPS reference missing
------------------------------------------------
If the Pulse Per Second (PPS) reference signal to any N200 device that Borealis will use upon startup
is not connected, the symptoms are:

    - Driver initialization doesn't proceed past initialization of the N200s.

*NOTE* This is as expected as the driver is waiting for a PPS signal to set the time registers

Start the radar by:

    - Ensure PPS signal is connected to each N200

Octoclock GPS Power loss
------------------------
If the master Octoclock (octoclock-g) unit loses power, then it no longer supplies 10MHz and PPS
reference signals to the slave Octoclocks. The symptoms are:

    - Octoclock slaves lose PPS and external 10MHz references (only the `power` LED is ON)
    - All `ref lock` front panel LEDs on all N200s are OFF
    - Continual lates from the driver (may take a few minutes for this symptom to manifest)

Start the radar by:

    - Ensure Octoclock-g has power connected, and GPS antenna is connected
    - `/borealis/stop_radar.sh`
    - `/borealis/start_radar.sh`
    - The driver will wait for GPS lock before initializing the N200s and starting the radar.

*NOTE* This may take a long time, and depends upon many factors including the antenna view of satellites, how long the
octoclock-g has been powered off, temperature, etc. In testing it locked within 20 minutes.

TXIO Cable disconnect from N200 or Transmitter
----------------------------------------------
If the cable carrying differential signals to/from the transmitters and the N200s is removed, or
has failed in some way, then some possible results are:

    - Transmitter will not transmit if the T/R signal is missing, this would be most obvious error
    - Transmitter Low Power and AGC Status signals may not be valid when read from the N200 GPIO
    - Transmitter may not be able to be placed into test mode

To fix this issue, ensure that all connectors are secured.

Shared memory full/Borealis unable to delete shared memory
----------------------------------------------------------
**NOTE** If you've just installed Borealis, this may be caused by a missing `h5copy` binary.
Make sure you have it installed for your operating system. For new versions of Ubuntu this means
installing `hdf5-tools`. For OpenSuSe it means installing `hdf5`.
If the shared memory location written to by Borealis is full, or the shared memory files are unable
to be deleted by Borealis, then some possible results are:

    - N200's will be in RX only mode (green LED on front panel will be on only)
    - Borealis will appear to halt when viewing the screen
    - Signal processing will quietly die
    - Data files, shared memory files and log files will cease being written

To fix this issue and restart the radar:
    - Make sure the `h5copy` binary is installed for your system
    - remove all Borealis created files in the `/dev/shm` directory
    - `/borealis/stop_radar.sh`
    - `/borealis/start_radar.sh`


remote_server.py Segfaults, other programs segfault (core-dump)
---------------------------------------------------------------
This behaviour has been seen several times at the Saskatoon Borealis radar.
The root cause is unknown, but symptoms are:

    - Radar stops with nothing obvious in the logs or on the screen session
    - Attempting to start the radar with `start_radar.sh` results in a segfault
    - Attempting to reboot the computer results in segfaults, bus errors, core dumps, etc
    
To fix this issue and restart the radar:
    - Power cycle the machine
