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
| site_id                        | sas                           | 3-letter standard ID of the radar     |
+--------------------------------+-------------------------------+---------------------------------------+
| gps_octoclock_addr             | addr=192.168.10.131           | IP address of the GPS Octoclock       |
+--------------------------------+-------------------------------+---------------------------------------+
| device_options                 | recv_frame_size=4000          | UHD USRP device arguments.            |
+--------------------------------+-------------------------------+---------------------------------------+
| main_antenna_count             | 16                            | Number of physical main array         |
|                                |                               | antennas, regardless of N200 status   |
+--------------------------------+-------------------------------+---------------------------------------+
| interferometer_antenna_count   | 4                             | Number of physical interferometer     |
|                                |                               | antennas, regardless of N200 status   |
+--------------------------------+-------------------------------+---------------------------------------+
| n200s                          | | {                           | List of all N200s, both active and    |
|                                | | addr : "192.168.10.100"     | not. Order of N200s is specified by   |
|                                | | rx : true                   | main_antenna, so N200s can be listed  |
|                                | | tx : true                   | out of order and more than the number |
|                                | | rx_int : false              | of antennas can be stored here (extra |
|                                | | main_antenna : "0"          | ones must have all flags set to       |
|                                | | interferometer_antenna : "" | false.) rx, tx, and rx_int set the    |
|                                | | }, {                        | receive, transmit, and interferometer |
|                                | | addr : "192.168.10.101"     | receive channels, respectively. The   |
|                                | | rx : true                   | antenna numbers refer to the physical |
|                                | | tx : false                  | antennas that are connected to the    |
|                                | | rx_int : true               | N200s. The number of main and         |
|                                | | main_antenna : "1"          | interferometer antennas that are      |
|                                | | interferometer_antenna : "0"| activated here must agree with the    |
|                                | | }, ...                      | count variables specified above.      |
|                                |                               | The ordering of the N200 parameters   |
|                                |                               | doesn't matter.                       |
+--------------------------------+-------------------------------+---------------------------------------+
| addr (n200s)                   | 192.168.10.100                | IP address of the specified N200      |
+--------------------------------+-------------------------------+---------------------------------------+
| rx (n200s)                     | true                          | Receive channel flag for a given      |
|                                |                               | N200. Set to true to activate receive |
|                                |                               | channel, and false to deactivate.     |
|                                |                               | If N200 is disconnected, this must    |
|                                |                               | be set to false.                      |
+--------------------------------+-------------------------------+---------------------------------------+
| tx (n200s)                     | true                          | Transmit channel flag for a given     |
|                                |                               | N200. Set to true to activate transmit|
|                                |                               | channel, and false to deactivate.     |
|                                |                               | If N200 is disconnected, this must    |
|                                |                               | be set to false.                      |
+--------------------------------+-------------------------------+---------------------------------------+
| rx_int (n200s)                 | false                         | Interferometer receive flag for a     |
|                                |                               | given N200. Set to true to activate   |
|                                |                               | the receive interferometer channel.   |
|                                |                               | A physical interferometer must be     |
|                                |                               | specified in int_antenna if this is   |
|                                |                               | set to true. If N200 is disconnected, |
|                                |                               | this must be set to false.            |
+--------------------------------+-------------------------------+---------------------------------------+
| main_antenna (n200s)           | 15                            | Physical antenna connected to this    |
|                                |                               | N200. N200s are sorted according to   |
|                                |                               | this value.                           |
+--------------------------------+-------------------------------+---------------------------------------+
| interterometer_antenna (n200s) | 3                             | Physical interferometer antenna       |
|                                |                               | connected to this N200. All antennas  |
|                                |                               | not connected must be set to empty    |
|                                |                               | string ("").                          |
+--------------------------------+-------------------------------+---------------------------------------+
| main_antenna_spacing           | 15.24                         | Distance between antennas (m).        |
+--------------------------------+-------------------------------+---------------------------------------+
| interferometer_antenna_spacing | 15.24                         | Distance between antennas (m).        |
+--------------------------------+-------------------------------+---------------------------------------+
| min_freq                       | 8.00E+06                      | Minimum frequency we can run (Hz).    |
+--------------------------------+-------------------------------+---------------------------------------+
| max_freq                       | 20.00E+06                     | Maximum frequency we can run (Hz).    |
+--------------------------------+-------------------------------+---------------------------------------+
| minimum_pulse_length           | 100                           | Minimum pulse length (us) dependent   |
|                                |                               | upon AGC feedback sample and hold.    |
+--------------------------------+-------------------------------+---------------------------------------+
| minimum_mpinc_length           | 1                             | Minimum length of multi-pulse         |
|                                |                               | increment (us).                       |
+--------------------------------+-------------------------------+---------------------------------------+
| minimum_pulse_separation       | 125                           | The minimum separation (us) before    |
|                                |                               | experiment treats it as a single      |
|                                |                               | pulse (transmitting zeroes and not    |
|                                |                               | receiving between the pulses. 125 us  |
|                                |                               | is approx two TX/RX times.            |
+--------------------------------+-------------------------------+---------------------------------------+
| tx_subdev                      | A:A                           | UHD daughterboard string which        |
|                                |                               | defines how to configure ports. Refer |
|                                |                               | to UHD subdev docs.                   |
+--------------------------------+-------------------------------+---------------------------------------+
| max_tx_sample_rate             | 5.00E+06                      | Maximum wideband TX rate each device  |
|                                |                               | can run in the system.                |
+--------------------------------+-------------------------------+---------------------------------------+
| main_rx_subdev                 | A:A A:B                       | UHD daughterboard string which        |
|                                |                               | defines how to configure ports. Refer |
|                                |                               | to UHD subdev docs.                   |
+--------------------------------+-------------------------------+---------------------------------------+
| interferometer_rx_subdev       | A:A A:B                       | UHD daughterboard string which        |
|                                |                               | defines how to configure ports. Refer |
|                                |                               | to UHD subdev docs.                   |
+--------------------------------+-------------------------------+---------------------------------------+
| max_rx_sample_rate             | 5.00E+06                      | Maximum wideband RX rate each         |
|                                |                               | device can run in the system.         |
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
| atr_rx                         | 0x0006                        | The pin mask for the RX only signal.  |
+--------------------------------+-------------------------------+---------------------------------------+
| atr_tx                         | 0x0018                        | The pin mask for the TX only signal.  |
+--------------------------------+-------------------------------+---------------------------------------+
| atr_xx                         | 0x0060                        | The pin mask for the full duplex      |
|                                |                               | signal (TR).                          |
+--------------------------------+-------------------------------+---------------------------------------+
| atr_0x                         | 0x0180                        | The pin mask for the idle signal.     |
+--------------------------------+-------------------------------+---------------------------------------+
| tst_md                         | 0x0600                        | The pin mask for the test mode signal |
+--------------------------------+-------------------------------+---------------------------------------+
| lo_pwr                         | 0x1800                        | The pin mask for the low power signal |
+--------------------------------+-------------------------------+---------------------------------------+
| agc_st                         | 0x6000                        | The pin mask for the AGC signal.      |
+--------------------------------+-------------------------------+---------------------------------------+
| tst_md                         | 0x6000                        | The pin mask for the test mode signal |
+--------------------------------+-------------------------------+---------------------------------------+
| max_usrp_dac_amplitude         | 0.99                          | The amplitude of highest allowed USRP |
|                                |                               | TX sample (V).                        |
+--------------------------------+-------------------------------+---------------------------------------+
| pulse_ramp_time                | 1.00E-05                      | The linear ramp time for the          |
|                                |                               | pulse (s)                             |
+--------------------------------+-------------------------------+---------------------------------------+
| tr_window_time                 | 6.00E-05                      | How much windowing on either side of  |
|                                |                               | pulse is needed for TR signal (s).    |
+--------------------------------+-------------------------------+---------------------------------------+
| agc_signal_read_delay          | 0                             | Hardware dependent delay after seq    |
|                                |                               | is finished for reading               |
|                                |                               | of AGC and low power signals (s)      |
+--------------------------------+-------------------------------+---------------------------------------+
| usrp_master_clock_rate         | 1.00E+08                      | Clock rate of the USRP master         |
|                                |                               | clock (Sps).                          |
+--------------------------------+-------------------------------+---------------------------------------+
| max_output_sample_rate         | 1.00E+05                      | Maximum rate allowed after            |
|                                |                               | downsampling (Sps)                    |
+--------------------------------+-------------------------------+---------------------------------------+
| max_number_of_filter_taps      | 2048                          | The maximum total number of filter    |
| _per_stage                     |                               | taps for all frequencies combined.    |
|                                |                               | This is a GPU limitation.             |
+--------------------------------+-------------------------------+---------------------------------------+
| router_address                 | tcp://127.0.0.1:6969          | The protocol/IP/port used for the ZMQ |
|                                |                               | router in Brian.                      |
+--------------------------------+-------------------------------+---------------------------------------+
| realtime_address               | tcp://eth0:9696               | The protocal/IP/port used for realtime|
+--------------------------------+-------------------------------+---------------------------------------+
| radctrl_to_exphan_identity     | RADCTRL_EXPHAN_IDEN           | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| radctrl_to_dsp_identity        | RADCTRL_DSP_IDEN              | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| radctrl_to_driver_identity     | RADCTRL_DRIVER_IDEN           | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| radctrl_to_brian_identity      | RADCTRL_BRIAN_IDEN            | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| radctrl_to_dw_identity         | RADCTRL_DW_IDEN               | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| driver_to_radctrl_identity     | DRIVER_RADCTRL_IDEN           | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| driver_to_dsp_identity         | DRIVER_DSP_IDEN               | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| driver_to_brian_identity       | DRIVER_BRIAN_IDEN             | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| driver_to_mainaffinity_identity| DRIVER_MAINAFFINITY_IDEN      | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| driver_to_txaffinity_identity  | DRIVER_TXAFFINITY_IDEN        | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| driver_to_rxaffinity_identity  | DRIVER_RXAFFINITY_IDEN        | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| mainaffinity_to_driver_identity| MAINAFFINITY_DRIVER_IDEN      | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| txaffinity_to_driver_identity  | TXAFFINITY_DRIVER_IDEN        | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| rxaffinity_to_driver_identity  | RXAFFINITY_DRIVER_IDEN        | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| exphan_to_radctrl_identity     | EXPHAN_RADCTRL_IDEN           | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| exphan_to_dsp_identity         | EXPHAN_DSP_IDEN               | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| dsp_to_radctrl_identity        | DSP_RADCTRL_IDEN              | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| dsp_to_driver_identity         | DSP_DRIVER_IDEN               | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| dsp_to_exphan_identity         | DSP_EXPHAN_IDEN               | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| dsp_to_dw_identity             | DSP_DW_IDEN                   | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| dspbegin_to_brian_identity     | DSPBEGIN_BRIAN_IDEN           | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| dspend_to_brian_identity       | DSPEND_BRIAN_IDEN             | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| dw_to_dsp_identity             | DW_DSP_IDEN                   | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| dw_to_radctrl_identity         | DW_RADCTRL_IDEN               | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| dw_to_rt_identity              | DW_RT_IDEN                    | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| rt_to_dw_identity              | RT_DW_IDEN                    | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| brian_to_radctrl_identity      | BRIAN_RADCTRL_IDEN            | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| brian_to_driver_identity       | BRIAN_DRIVER_IDEN             | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| brian_to_dspbegin_identity     | BRIAN_DSPBEGIN_IDEN           | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| brian_to_dspend_identity       | BRIAN_DSPEND_IDEN             | ZMQ named socket identity.            |
+--------------------------------+-------------------------------+---------------------------------------+
| ringbuffer_name                | data_ringbuffer               | Shared memory name for ringbuffer.    |
+--------------------------------+-------------------------------+---------------------------------------+
| ringbuffer_size_bytes          | 200.00E+06                    | Size in bytes to allocate for each    |
|                                |                               | ringbuffer.                           |
+--------------------------------+-------------------------------+---------------------------------------+
| data_directory                 | /data/borealis_data           | Location of output data files.        |
+--------------------------------+-------------------------------+---------------------------------------+
| log_directory                  | /data/borealis_logs           | Location of output log files          |
+--------------------------------+-------------------------------+---------------------------------------+
| hdw_path                       | /usr/local/hdw                | Path to locally cloned SuperDARN      |
|                                |                               | hardware repository                   |
+--------------------------------+-------------------------------+---------------------------------------+

----------------------
Example configurations
----------------------
There are several instances when you'll need to modify this file for correct operation.

#. One of your main array antennas is not working properly (broken coax, blown lightning arrestor,
   etc)

    The rx and tx flags for the associated N200 should be set to false. This will disable the
    receive and transmit channels, and stop the N200s from collecting samples from that antenna.
    Note: If the N200 is also connected to an interferometer antenna, the interferometer antenna
    will also have to be disconnected by setting rx_int to false, or moving it to a different N200.

#. One of your interferometer array antennas is not working properly (broken coax, blown lightning
   arrestor, etc)

    The rx_int flag for the associated N200 should be set to false. This will disable the
    interferometer receive channel for that antenna, and stop the N200s from collecting samples from
    that antenna.

#. One of your transmitter's transmit paths is not working, but the receive path is still working
   properly

    The tx flag for the associated N200 should be set to false. This will disable the transmission
    channel on the bad transmit path. **Note: This configuration does not work with the current
    iteration of Borealis**

#. One of your transmitter's receive paths is not working, but the transmit path is still working
   properly

    The rx flag for the associated N200 should be set to false. This will disable the receive
    channel on the bad receive path. **Note: This configuration does not work with the current
    iteration of Borealis**

#. One of your transmitters is not working at all

    The rx and tx flags for the N200 connected to the non-working transmitter should both be set to
    false. This will disable the transmit and receive channels for that transmitter.

#. One of your N200s is not working properly and you've inserted the spare N200

    Add an entry for the replacement N200, and copy rx, tx, rx_int, main_antenna, and
    interferometer_antenna from the broken N200. Set all the flags for the broken N200 to false, and
    set main_antenna and interferometer_antenna to empty strings to deactivate the N200. The entry
    for the broken N200 can be left in the config file for future use, as the code will ignore the
    broken N200 and replace it with the new one.

#. One of your N200s is not working properly but you're located remotely and cannot insert the spare
   N200

    This particular N200 will have to be deactivated. To do this, set all flags to false (tx, rx,
    and rx_int).

#. You have a non-standard array

    One example of a non-standard array would be a different number of interferometer antennas than
    four. To implement this, modify the individual N200 entries to specify which N200s are connected
    to interferometer antennas. Additionally, set the main and interferometer antenna count
    parameters to the number of physical antennas in the array.

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

    #. ``n200s`` - Set tx, rx, and rx_int flags to true for only one N200, all other N200s should
       have their flags set to false.

    #. ``pps`` and ``ref`` - These should both be set to ``internal``, as you don't have an
       Octoclock to provide a reference PPS or 10MHz reference signal.
