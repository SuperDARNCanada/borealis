.. _config-options:

=============
Configuration
=============

-----------------
Config Parameters
-----------------
+--------------------------------+-------------------------------+---------------------------------------+
| Config field                   | Example entry                 | Description                           |
+================================+===============================+=======================================+
| site_id                        | sas                           | 3-letter standard ID of the radar.    |
+--------------------------------+-------------------------------+---------------------------------------+
| gps_octoclock_addr             | addr=192.168.10.131           | IP address of the GPS Octoclock.      |
+--------------------------------+-------------------------------+---------------------------------------+
| device_options                 | recv_frame_size=4000          | UHD USRP device arguments.            |
+--------------------------------+-------------------------------+---------------------------------------+
| main_antenna_count             | 16                            | Number of physical main array         |
|                                |                               | antennas, regardless of N200 status.  |
+--------------------------------+-------------------------------+---------------------------------------+
| intf_antenna_count             | 4                             | Number of physical interferometer     |
|                                |                               | antennas, regardless of N200 status.  |
+--------------------------------+-------------------------------+---------------------------------------+
| n200s                          | | {                           | List of all N200s, both active and    |
|                                | | "addr" : "192.168.10.100",  | not. The value for each channel maps  |
|                                | | "rx_channel_0" : "m0",      | to a physical antenna of the radar,   |
|                                | | "rx_channel_1" : "i0",      | with the first digit either "m" for   |
|                                | | "tx_channel_0" : "m0"       | main array or "i" for interferometer  |
|                                | | }, {                        | array. TX channels can only be        |
|                                | | "addr" : "192.168.10.101",  | connected to the main array. The      |
|                                | | "rx_channel_0" : "m1",      | numbers after the array designator    |
|                                | | "rx_channel_1" : "i1",      | are the antenna index into the array. |
|                                | | "tx_channel_0" : "m1"       | The antenna index must lie between 0  |
|                                | | }, ...                      | and the [main|intf]_antenna_count     |
|                                |                               | fields above. The ordering of the     |
|                                |                               | N200 parameters doesn't matter.       |
+--------------------------------+-------------------------------+---------------------------------------+
| addr (n200s)                   | 192.168.10.100                | IP address of the specified N200.     |
+--------------------------------+-------------------------------+---------------------------------------+
| rx_channel_0 (n200s)           | m0                            | Antenna number connected to receive   |
|                                |                               | channel 0 of the N200. The first      |
|                                |                               | character indicates the array, and    |
|                                |                               | the rest the index of the antenna.    |
|                                |                               | Set to "" if the channel is           |
|                                |                               | disconnected.                         |
+--------------------------------+-------------------------------+---------------------------------------+
| rx_channel_1 (n200s)           | i0                            | Antenna number connected to receive   |
|                                |                               | channel 1 of the N200. The first      |
|                                |                               | character indicates the array, and    |
|                                |                               | the rest the index of the antenna.    |
|                                |                               | Set to "" if the channel is           |
|                                |                               | disconnected.                         |
+--------------------------------+-------------------------------+---------------------------------------+
| tx_channel_0 (n200s)           | m0                            | Antenna number connected to transmit  |
|                                |                               | channel 0 of the N200. The first      |
|                                |                               | character indicates the array, and    |
|                                |                               | the rest the index of the antenna.    |
|                                |                               | The array specifier must be "m", if   |
|                                |                               | set. Set to "" if the channel is      |
|                                |                               | disconnected.                         |
+--------------------------------+-------------------------------+---------------------------------------+
| main_antenna_spacing           | 15.24                         | Distance between antennas (m).        |
+--------------------------------+-------------------------------+---------------------------------------+
| intf_antenna_spacing           | 15.24                         | Distance between antennas (m).        |
+--------------------------------+-------------------------------+---------------------------------------+
| min_freq                       | 8.00E+06                      | Minimum frequency we can run (Hz).    |
+--------------------------------+-------------------------------+---------------------------------------+
| max_freq                       | 20.00E+06                     | Maximum frequency we can run (Hz).    |
+--------------------------------+-------------------------------+---------------------------------------+
| min_pulse_length               | 100                           | Minimum pulse length (us) dependent   |
|                                |                               | upon AGC feedback sample and hold.    |
+--------------------------------+-------------------------------+---------------------------------------+
| min_tau_spacing_length         | 1                             | Minimum length of multi-pulse         |
|                                |                               | increment (us).                       |
+--------------------------------+-------------------------------+---------------------------------------+
| min_pulse_separation           | 125                           | The minimum separation (us) before    |
|                                |                               | experiment treats it as a single      |
|                                |                               | pulse (transmitting zeroes and not    |
|                                |                               | receiving between the pulses). 125 us |
|                                |                               | is approx two TX/RX times.            |
+--------------------------------+-------------------------------+---------------------------------------+
| max_tx_sample_rate             | 5.00E+06                      | Maximum wideband TX rate each device  |
|                                |                               | can run in the system.                |
+--------------------------------+-------------------------------+---------------------------------------+
| max_rx_sample_rate             | 5.00E+06                      | Maximum wideband RX rate each         |
|                                |                               | device can run in the system.         |
+--------------------------------+-------------------------------+---------------------------------------+
| tx_subdev                      | A:A                           | UHD daughterboard string which        |
|                                |                               | defines how to configure ports. Refer |
|                                |                               | to UHD subdev docs.                   |
+--------------------------------+-------------------------------+---------------------------------------+
| main_rx_subdev                 | A:A A:B                       | UHD daughterboard string which        |
|                                |                               | defines how to configure ports. Refer |
|                                |                               | to UHD subdev docs.                   |
+--------------------------------+-------------------------------+---------------------------------------+
| intf_rx_subdev                 | A:A A:B                       | UHD daughterboard string which        |
|                                |                               | defines how to configure ports. Refer |
|                                |                               | to UHD subdev docs.                   |
+--------------------------------+-------------------------------+---------------------------------------+
| pps                            | external                      | The PPS source for the system         |
|                                |                               | (internal, external, none).           |
+--------------------------------+-------------------------------+---------------------------------------+
| ref                            | external                      | The 10 MHz reference source           |
|                                |                               | (internal, external).                 |
+--------------------------------+-------------------------------+---------------------------------------+
| overthewire                    | sc16                          | Data type for samples the USRP        |
|                                |                               | operates with. Refer to UHD docs for  |
|                                |                               | data types.                           |
+--------------------------------+-------------------------------+---------------------------------------+
| cpu                            | fc32                          | Data type of samples that UHD uses    |
|                                |                               | on host CPU. Refer to UHD docs for    |
|                                |                               | data types.                           |
+--------------------------------+-------------------------------+---------------------------------------+
| gpio_bank_high                 | RXA                           | The daughterboard pin bank to use for |
|                                |                               | active-high TR and I/O signals.       |
+--------------------------------+-------------------------------+---------------------------------------+
| gpio_bank_low                  | TXA                           | The daughterboard pin bank to use for |
|                                |                               | active-low TR and I/O signals.        |
+--------------------------------+-------------------------------+---------------------------------------+
| atr_rx                         | 0x0006                        | The pin mask for the RX-only signal.  |
+--------------------------------+-------------------------------+---------------------------------------+
| atr_tx                         | 0x0018                        | The pin mask for the TX-only signal.  |
+--------------------------------+-------------------------------+---------------------------------------+
| atr_xx                         | 0x0060                        | The pin mask for the full-duplex      |
|                                |                               | signal (TR).                          |
+--------------------------------+-------------------------------+---------------------------------------+
| atr_0x                         | 0x0180                        | The pin mask for the idle signal.     |
+--------------------------------+-------------------------------+---------------------------------------+
| lo_pwr                         | 0x0600                        | The pin mask for the low-power signal.|
+--------------------------------+-------------------------------+---------------------------------------+
| agc_st                         | 0x1800                        | The pin mask for the AGC signal.      |
+--------------------------------+-------------------------------+---------------------------------------+
| tst_md                         | 0x6000                        | The pin mask for the test mode signal.|
+--------------------------------+-------------------------------+---------------------------------------+
| max_usrp_dac_amplitude         | 0.99                          | Amplitude of the highest allowed USRP |
|                                |                               | TX sample (V).                        |
+--------------------------------+-------------------------------+---------------------------------------+
| pulse_ramp_time                | 1.00E-05                      | The linear ramp time for the          |
|                                |                               | pulse (s).                            |
+--------------------------------+-------------------------------+---------------------------------------+
| tr_window_time                 | 6.00E-05                      | How much windowing on either side of  |
|                                |                               | pulse is needed for TR signal (s).    |
+--------------------------------+-------------------------------+---------------------------------------+
| agc_signal_read_delay          | 0                             | Hardware dependent delay after seq    |
|                                |                               | is finished for reading               |
|                                |                               | of AGC and low power signals (s).     |
+--------------------------------+-------------------------------+---------------------------------------+
| usrp_master_clock_rate         | 1.00E+08                      | Clock rate of the USRP master         |
|                                |                               | clock (Sps).                          |
+--------------------------------+-------------------------------+---------------------------------------+
| max_output_sample_rate         | 1.00E+05                      | Maximum rate allowed after            |
|                                |                               | downsampling (Sps).                   |
+--------------------------------+-------------------------------+---------------------------------------+
| max_filtering_stages           | 6                             | The maximum number of filtering       |
|                                |                               | stages.                               |
+--------------------------------+-------------------------------+---------------------------------------+
| max_filter_taps_per_stage      | 2048                          | The maximum total number of filter    |
|                                |                               | taps for all frequencies combined.    |
|                                |                               | This is a GPU limitation.             |
+--------------------------------+-------------------------------+---------------------------------------+
| router_address                 | tcp://127.0.0.1:6969          | The protocol/IP/port used for the ZMQ |
|                                |                               | router in Brian.                      |
+--------------------------------+-------------------------------+---------------------------------------+
| realtime_address               | tcp://eth0:9696               | The protocol/IP/port used for         |
|                                |                               | realtime.                             |
+--------------------------------+-------------------------------+---------------------------------------+
| ringbuffer_name                | data_ringbuffer               | Shared memory name for ringbuffer.    |
+--------------------------------+-------------------------------+---------------------------------------+
| ringbuffer_size_bytes          | 200.00E+06                    | Size in bytes to allocate for each    |
|                                |                               | ringbuffer.                           |
+--------------------------------+-------------------------------+---------------------------------------+
| data_directory                 | /data/borealis_data           | Location of output data files.        |
+--------------------------------+-------------------------------+---------------------------------------+
| log_handlers                   | | {                           | Fields for controlling the Borealis   |
|                                | | "console" : {},             | loggers. Supported log handlers are   |
|                                | | "logfile" : {},             | console logging, JSON file logging,   |
|                                | | "aggregator" : {},          | and aggregator log forwarding.        |
|                                | | }                           |                                       |
+--------------------------------+-------------------------------+---------------------------------------+
| console (log_handlers)         | | {                           | An enable flag and log level for      |
|                                | | "enable" : true,            | the console log handler. Supported    |
|                                | | "level" : "INFO"            | levels are "DEBUG", "VERBOSE",        |
|                                | | }                           | "INFO", "WARNING", "ERROR", "NOTSET", |
|                                |                               | and "CRITICAL", or a numeric value    |
|                                |                               | between 0 and 50.                     |
+--------------------------------+-------------------------------+---------------------------------------+
| logfile (log_handlers)         | | {                           | An enable flag, log level, and        |
|                                | | "enable" : true,            | path to a directory for storing log   |
|                                | | "level" : "VERBOSE",        | files in. The log levels are the same |
|                                | | "directory" :               | as for ``console`` above.             |
|                                | |      "/data/borealis_logs"  |                                       |
|                                | | }                           |                                       |
+--------------------------------+-------------------------------+---------------------------------------+
| aggregator (log_handlers)      | | {                           | An enable flag, log level, and        |
|                                | | "enable" : true,            | network address and port for          |
|                                | | "level" : "INFO",           | aggregator log handling. The logs are |
|                                | | "addr" : "0.0.0.0",         | then sent over the network for        |
|                                | | "port" : "12201"            | collection by an aggregator such as   |
|                                | | }                           | a graylog server. The log levels are  |
|                                |                               | the same as for ``console`` above.    |
+--------------------------------+-------------------------------+---------------------------------------+
| hdw_path                       | /usr/local/hdw                | Path to locally cloned SuperDARN      |
|                                |                               | hardware repository.                  |
+--------------------------------+-------------------------------+---------------------------------------+

---------------------
Testing a config file
---------------------
A test script is available for verifying the fields of your configuration file. This script is detailed further
:ref:`here. <Config Testing>`

----------------------
Example configurations
----------------------
There are several instances when you'll need to modify this file for correct operation.

#. One of your main array antennas is not working properly (broken coax, blown lightning arrestor,
   etc)

    The N200(s) with RX and TX channels connected to that antenna should have those channels set to ``""``.
    This will disable transmission and reception on the antenna, while preserving the correct phasing
    for beamforming on other channels.

#. One of your interferometer array antennas is not working properly (broken coax, blown lightning
   arrestor, etc)

    The N200 with an RX channel connected to that antenna should have that channel set to ``""``.
    This will disable reception from the antenna, while preserving the correct phasing for beamforming
    data from other channels.

#. One of your transmitter's transmit paths is not working, but the receive path is still working
   properly

    The ``tx_channel_0`` field for the associated N200 should be set to ``""``. This will disable the transmission
    channel on the bad transmit path.

#. One of your transmitter's receive paths is not working, but the transmit path is still working
   properly

    The ``rx_channel_#`` flag for the associated N200 should be set to ``""``. This will disable the receive
    channel on the bad receive path.

#. One of your transmitters is not working at all

    Ensure that no N200s have a channel set to use that corresponding antenna. For example, if the transmitter
    for antenna 7 of the main array is broken, make sure no ``rx_channel_#`` or ``tx_channel_0`` fields are set
    to ``"m7"``.

#. One of your N200s is not working properly and you've inserted the spare N200

    Add an entry for the replacement N200, and copy ``rx_channel_0``, ``rx_channel_1``, and ``tx_channel_0``
    fields from the broken N200. Make sure that the cables are transferred over, and verify that the cabling
    matches with the antennas specified for each channel. The configuration for the broken N200 can be set to ``""``
    for each channel. If all channels are set to ``""``, the N200 is ignored.

#. One of your N200s is not working properly but you're located remotely and cannot insert the spare
   N200

    This particular N200 will have to be deactivated. To do this, set all channel fields to ``""``.

#. You have a non-standard array

    One example of a non-standard array would be a different number of interferometer antennas than
    four. To implement this, modify the individual N200 entries to specify which N200s are connected
    to interferometer antennas. Additionally, set the main and interferometer antenna count
    parameters to the number of physical antennas in each array.

#. You want to change the location of ATR signals on the daughterboards

    This can be done by changing the values of the following config parameters: atr_rx, atr_tx,
    atr_xx, atr_0x, tst_md, lo_pwr, agc_st. The value ``atr_rx = 0x0006`` means that the ATR_RX
    signal will appear on the pins 1 and 2 (referenced from 0). I.e. every bit that is a '1' in this
    hex value indicates which pin the signal will appear on.

#. You want to change the polarity of the ATR signals on the daughterboards

    This can be done by swapping the values of the two config parameters: ``gpio_bank_high`` and
    ``gpio_bank_low``. The default is for active-high signals to be on the LFRX daughterboard. This
    is done by setting ``gpio_bank_high`` to ``RXA``. The same signals, but active-low, are by
    default located on the LFTX daughterboard.

#. You would like to make a test-system with only one N200 and don't have any Octoclocks

    This can be done by changing the following parameters:

    #. ``n200s`` - Set ``tx_channel_0``, ``rx_channel_0``, and ``rx_channel_1`` fields for only one N200. All
       others should have their channels set to ``""``.

    #. ``pps`` and ``ref`` - These should both be set to ``internal``, as you don't have an
       Octoclock to provide a reference PPS or 10MHz reference signal.
