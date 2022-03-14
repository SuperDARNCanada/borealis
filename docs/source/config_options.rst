*****************
Config Parameters
*****************
+-----------------------------------------+-------------------------+--------------------------------------+
|Config field                             | Example entry           | Description                          |
+=========================================+=========================+======================================+
| site_id                                 | sas                     | 3-letter standard ID of the radar    |
+-----------------------------------------+-------------------------+--------------------------------------+
| gps_octoclock_addr                      | addr=192.168.10.131     | IP address of the GPS Octoclock      |
+-----------------------------------------+-------------------------+--------------------------------------+
| devices                                 | recv_frame_size=4000,   | UHD USRP device arguments.           |
|                                         | addr0=192.168.10.100,   |                                      |
|                                         | addr1=192.168.10.101,   |                                      |
|                                         | addr2=192.168.10.102,   |                                      |
|                                         | addr3=192.168.10.103,   |                                      |
|                                         | addr4=192.168.10.104,   |                                      |
|                                         | addr5=192.168.10.105,   |                                      |
|                                         | addr6=192.168.10.106,   |                                      |
|                                         | addr7=192.168.10.107,   |                                      |
|                                         | addr8=192.168.10.108,   |                                      |
|                                         | addr9=192.168.10.109,   |                                      |
|                                         | addr10=192.168.10.110,  |                                      |
|                                         | addr11=192.168.10.111,  |                                      |
|                                         | addr12=192.168.10.112,  |                                      |
|                                         | addr13=192.168.10.113,  |                                      |
|                                         | addr14=192.168.10.114,  |                                      |
|                                         | addr15=192.168.10.115   |                                      |
+-----------------------------------------+-------------------------+--------------------------------------+
| main_antennas                           | 0,1,2,3,4,5,6,7,8,9,10, | Mapping of main rx/tx channels to    |
|                                         | 11,12,13,14,15          | antennas in the main array           |
+-----------------------------------------+-------------------------+--------------------------------------+
| main_antenna_count                      | 16                      | Number of physical main array        |
|                                         |                         | antennas                             |
+-----------------------------------------+-------------------------+--------------------------------------+
| interferometer_antennas                 | 0,1,2,3                 | Mapping of intf rx channels to       |
|                                         |                         | antennas in the interferometer array |
+-----------------------------------------+-------------------------+--------------------------------------+
| interferometer_antenna_count            | 4                       | Number of physical interferometer    |
|                                         |                         | antennas                             |
+-----------------------------------------+-------------------------+--------------------------------------+
| main_antenna_usrp_rx_channels           | 0,2,4,6,8,10,12,14,16,  | UHD channel designation for RX main  |
|                                         | 18,20,22,24,26,28,30    | antennas                             |
+-----------------------------------------+-------------------------+--------------------------------------+
| interferometer_antenna_usrp_rx_channels | 1,3,5,7                 | UHD channel designation for RX intf  |
|                                         |                         | antennas.                            |
+-----------------------------------------+-------------------------+--------------------------------------+
| main_antenna_usrp_tx_channels           | 0,1,2,3,4,5,6,7,8,9,    | UHD channel designation for TX main  |
|                                         | 10,11,12,13,14,15       | antennas.                            |
+-----------------------------------------+-------------------------+--------------------------------------+
| main_antenna_spacing                    | 15.24                   | Distance between antennas (m).       |
+-----------------------------------------+-------------------------+--------------------------------------+
| interferometer_antenna_spacing          | 15.24                   | Distance between antennas (m).       |
+-----------------------------------------+-------------------------+--------------------------------------+
| min_freq                                | 8.00E+06                | Minimum frequency we can run (Hz).   |
+-----------------------------------------+-------------------------+--------------------------------------+
| max_freq                                | 20.00E+06               | Maximum frequency we can run (Hz).   |
+-----------------------------------------+-------------------------+--------------------------------------+
| minimum_pulse_length                    | 100                     | Minimum pulse length (us) dependent  |
|                                         |                         | upon AGC feedback sample and hold.   |
+-----------------------------------------+-------------------------+--------------------------------------+
| minimum_mpinc_length                    | 1                       | Minimum length of multi-pulse        |
|                                         |                         | increment (us).                      |
+-----------------------------------------+-------------------------+--------------------------------------+
| minimum_pulse_separation                | 125                     | The minimum separation (us) before   |
|                                         |                         | experiment treats it as a single     |
|                                         |                         | pulse (transmitting zeroes and not   |
|                                         |                         | receiving between the pulses. 125 us |
|                                         |                         | is approx two TX/RX times.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| tx_subdev                               | A:A                     | UHD daughterboard string which       |
|                                         |                         | defines how to configure ports. Refer|
|                                         |                         | to UHD subdev docs.                  |
+-----------------------------------------+-------------------------+--------------------------------------+
| max_tx_sample_rate                      | 5.00E+06                | Maximum wideband TX rate each device |
|                                         |                         | can run in the system.               |
+-----------------------------------------+-------------------------+--------------------------------------+
| main_rx_subdev                          | A:A A:B                 | UHD daughterboard string which       |
|                                         |                         | defines how to configure ports. Refer|
|                                         |                         | to UHD subdev docs.                  |
+-----------------------------------------+-------------------------+--------------------------------------+
| interferometer_rx_subdev                | A:A A:B                 | UHD daughterboard string which       |
|                                         |                         | defines how to configure ports. Refer|
|                                         |                         | to UHD subdev docs.                  |
+-----------------------------------------+-------------------------+--------------------------------------+
| max_rx_sample_rate                      | 5.00E+06                | Maximum wideband RX rate each        |
|                                         |                         | device can run in the system.        |
+-----------------------------------------+-------------------------+--------------------------------------+
| pps                                     | external                | The PPS source for the system        |
|                                         |                         | (internal, external, none).          |
+-----------------------------------------+-------------------------+--------------------------------------+
| ref                                     | external                | The 10 MHz reference source          |
|                                         |                         | (internal, external).                |
+-----------------------------------------+-------------------------+--------------------------------------+
| overthewire                             | sc16                    | Data type for samples the USRP       |
|                                         |                         | operates with. Refer to UHD docs for |
|                                         |                         | data types.                          |
+-----------------------------------------+-------------------------+--------------------------------------+
| cpu                                     | fc32                    | Data type of samples that UHD uses   |
|                                         |                         | on host CPU. Refer to UHD docs for   |
|                                         |                         | data types.                          |
+-----------------------------------------+-------------------------+--------------------------------------+
| gpio_bank_high                          | RXA                     | The daughterboard pin bank to use for|
|                                         |                         | active-high TR and I/O signals.      |
+-----------------------------------------+-------------------------+--------------------------------------+
| gpio_bank_low                           | TXA                     | The daughterboard pin bank to use for|
|                                         |                         | active-low TR and I/O signals.       |
+-----------------------------------------+-------------------------+--------------------------------------+
| atr_rx                                  | 0x0006                  | The pin mask for the RX only mode.   |
+-----------------------------------------+-------------------------+--------------------------------------+
| atr_tx                                  | 0x0018                  | The pin mask for the TX only mode.   |
+-----------------------------------------+-------------------------+--------------------------------------+
| atr_xx                                  | 0x0060                  | The pin mask for the full duplex     |
|                                         |                         | mode (TR).                           |
+-----------------------------------------+-------------------------+--------------------------------------+
| atr_0x                                  | 0x0180                  | The pin mask for the idle mode.      |
+-----------------------------------------+-------------------------+--------------------------------------+
| tst_md                                  | 0x0600                  | The pin mask for test mode.          |
+-----------------------------------------+-------------------------+--------------------------------------+
| lo_pwr                                  | 0x1800                  | The pin mask for the low power signal|
+-----------------------------------------+-------------------------+--------------------------------------+
| agc_st                                  | 0x6000                  | The pin mask for the AGC signal.     |
+-----------------------------------------+-------------------------+--------------------------------------+
| max_usrp_dac_amplitude                  | 0.99                    | The amplitude of highest allowed USRP|
|                                         |                         | TX sample (V).                       |
+-----------------------------------------+-------------------------+--------------------------------------+
| pulse_ramp_time                         | 1.00E-05                | The linear ramp time for the         |
|                                         |                         | pulse (s)                            |
+-----------------------------------------+-------------------------+--------------------------------------+
| tr_window_time                          | 6.00E-05                | How much windowing on either side of |
|                                         |                         | pulse is needed for TR signal (s).   |
+-----------------------------------------+-------------------------+--------------------------------------+
| agc_signal_read_delay                   | 0                       | Hardware dependent delay time for    |
|                                         |                         | reading of AGC and low power signals |
+-----------------------------------------+-------------------------+--------------------------------------+
| usrp_master_clock_rate                  | 1.00E+08                | Clock rate of the USRP master        |
|                                         |                         | clock (Sps).                         |
+-----------------------------------------+-------------------------+--------------------------------------+
| max_output_sample_rate                  | 1.00E+05                | Maximum rate allowed after           |
|                                         |                         | downsampling (Sps)                   |
+-----------------------------------------+-------------------------+--------------------------------------+
| max_number_of_filter_taps_per_stage     | 2048                    | The maximum total number of filter   |
|                                         |                         | taps for all frequencies combined.   |
|                                         |                         | This is a GPU limitation.            |
+-----------------------------------------+-------------------------+--------------------------------------+
| router_address                          | tcp://127.0.0.1:6969    | The protocol/IP/port used for the ZMQ|
|                                         |                         | router in Brian.                     |
+-----------------------------------------+-------------------------+--------------------------------------+
| radctrl_to_exphan_identity              | RADCTRL_EXPHAN_IDEN     | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| radctrl_to_dsp_identity                 | RADCTRL_DSP_IDEN        | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| radctrl_to_driver_identity              | RADCTRL_DRIVER_IDEN     | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| radctrl_to_brian_identity               | RADCTRL_BRIAN_IDEN      | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| radctrl_to_dw_identity                  | RADCTRL_DW_IDEN         | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| driver_to_radctrl_identity              | DRIVER_RADCTRL_IDEN     | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| driver_to_dsp_identity                  | DRIVER_DSP_IDEN         | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| driver_to_brian_identity                | DRIVER_BRIAN_IDEN       | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| exphan_to_radctrl_identity              | EXPHAN_RADCTRL_IDEN     | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| exphan_to_dsp_identity                  | EXPHAN_DSP_IDEN         | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| dsp_to_radctrl_identity                 | DSP_RADCTRL_IDEN        | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| dsp_to_driver_identity                  | DSP_DRIVER_IDEN         | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| dsp_to_exphan_identity                  | DSP_EXPHAN_IDEN         | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| dsp_to_dw_identity                      | DSP_DW_IDEN             | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| dspbegin_to_brian_identity              | DSPBEGIN_BRIAN_IDEN     | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| dspend_to_brian_identity                | DSPEND_BRIAN_IDEN       | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| dw_to_dsp_identity                      | DW_DSP_IDEN             | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| dw_to_radctrl_identity                  | DW_RADCTRL_IDEN         | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| brian_to_radctrl_identity               | BRIAN_RADCTRL_IDEN      | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| brian_to_driver_identity                | BRIAN_DRIVER_IDEN       | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| brian_to_dspbegin_identity              | BRIAN_DSPBEGIN_IDEN     | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| brian_to_dspend_identity                | BRIAN_DSPEND_IDEN       | ZMQ named socket identity.           |
+-----------------------------------------+-------------------------+--------------------------------------+
| ringbuffer_name                         | data_ringbuffer         | Shared memory name for ringbuffer.   |
+-----------------------------------------+-------------------------+--------------------------------------+
| ringbuffer_size_bytes                   | 200.00E+06              | Size in bytes to allocate for each   |
|                                         |                         | ringbuffer.                          |
+-----------------------------------------+-------------------------+--------------------------------------+
| data_directory                          | /data/borealis_data     | Location of output data files.       |
+-----------------------------------------+-------------------------+--------------------------------------+

**********************
Example configurations
**********************
There are several instances when you'll need to modify this file for correct operation.

#. One of your main array antennas is not working properly (broken coax, blown lightning arrestor, etc)

    In this situation, you have two options:

    #. Leave the N200 running and collecting data. This antenna will not transmit or receive properly,
        so the data collected will bring down the signal strength in the bfiq data as one antenna will
        essentially be measuring noise.

    #. Remove the N200 from operation. Follow the steps for a broken N200 with no replacement below.

#. One of your interferometer array antennas is not working properly (broken coax, blown lightning arrestor, etc)

    In this situation, you have two options:

    #. Receive data on the affected channel. This will skew the bfiq data (and rawacf) for the array, as this channel
        will essentially be noise, averaged with the signals from the other antennas when processed into bfiq.

    #. Do not receive data on the channel. This can be done by changing the following:

       * `interferometer_antennas` - remove the index of the affected antenna from the list.

       * `interferometer_antenna_usrp_rx_channels` - remove the channel of the affected antenna from the list.

#. One of your transmitter's transmit paths is not working, but the receive path is still working properly

TODO

#. One of your transmitter's receive paths is not working, but the transmit path is still working properly

TODO

#. One of your transmitters is not working at all

TODO

#. One of your N200s is not working properly and you've inserted the spare N200

    In this instance, since you still have the same number of antennas as well as transmit and receive channels,
    you simply need to change the IP adress of the N200 you replaced. This is done in the `devices` config option.
    An example: if N200 with IP address 192.168.10.104 dies, and is replaced with the spare (ip address 192.168.10.116),
    simply replace `addr4=192.168.10.104` with `addr4=192.168.10.116`.

#. One of your N200s is not working properly but you're located remotely and cannot insert the spare N200

    #. Remove the corresponding address from the `devices` field, and shift the remaining IP addresses (cannot have a
        gap like `addr0=xxx,addr2=xxx).

    #. Remove the corresponding main antenna index from the `main_antennas` field.

    #. If the N200 is also receiving an interferometer channel, remove the interferometer index from the
        `interferometer_antennas` field.

    #. Remove the last rx channel from `main_antenna_usrp_rx_channels`. These indices map to the `devices` list, with
        each N200 having two rx channels. This means rx channels 0 and 1 map to `addr0` in `devices`, channels 2 and 3
        to `addr1`, and so on. The same applies to tx channels, however each N200 only has 1.

    #. If applicable, remove the channel from `interferometer_antenna_usrp_rx_channels` that corresponds to the rx
        channel on the removed device.

    #. Remove the last tx channel from `main_antenna_usrp_tx_channels`. This is done for the same reasons as removing
        the last rx channel in `main_antenna_usrp_rx_channels` above.

    To illustrate, let's consider the case where N200 `192.168.10.103` goes down, assuming your config.ini nominally has
    the same fields as the table at the top of the page. The fields should now read:

    * `"devices" : "...,addr2=192.168.10.102,addr3=192.168.10.104,addr4=192.168.10.105,..."`
    * `"main_antennas" : "0,1,2,4,5,..."`
    * `"interferometer_antennas" : "0,1,2"`
    * `"main_antenna_usrp_rx_channels" : "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28"`
    * `"interferometer_antenna_usrp_rx_channels" : "1,3,5"`
    * `"main_antenna_usrp_tx_channels" : "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"`

#. You have a non-standard array

    One example of a non-standard array would be a different number of interferometer antennas than four.
    If your interferometer array has only two antennas you'll need to modify the following:

    #. interferometer_antennas = 0,1

    #. interferometer_antenna_count = 2

    #. interferometer_antenna_usrp_rx_channels = 1,3

#. You want to change the location of ATR signals on the daughterboards

    This can be done by changing the values of the following config parameters:
    atr_rx, atr_tx, atr_xx, atr_0x, tst_md, lo_pwr, agc_st.
    The value `atr_rx = 0x0006` means that the ATR_RX signal will appear on the pins 1 and 2 (referenced from 0). I.e. every bit that is a '1' in this hex value indicates which pin the signal will appear on.

#. You want to change the polarity of the ATR signals on the daughterboards

    This can be done by swapping the values of the two config parameters: `gpio_bank_high` and `gpio_bank_low`.
    The default is for active-high signals to be on the LFRX daughterboard. This is done by setting `gpio_bank_high` to `RXA`.
    The same signals, but active-low, are by default located on the LFTX daughterboard.

#. You would like to make a test-system with only one N200 and don't have any Octoclocks

    This can be done by changing the following parameters:

    #. `devices` - Should only have one address (addr0=192.168.10.xxx)

    #. `main_antenna_count` - If you only have one N200, this should be set to 1, as there is only one transmit channel per N200.

    #. `interferometer_antenna_count` - With only one N200, this should be set to 0 or 1.

    #. `main_antenna_usrp_channels` - There will only be two rx channels available, so this should be a single element, and it should be `0`

    #. `interferometer_antenna_usrp_rx_channels` - The second rx channel available should be placed here, so it will be `1`

    #. `main_antenna_usrp_tx_channels` - As discussed above, only one transmit channel exists, so this should be set to `0`

    #. `pps` and `ref` - These should both be set to `internal`, as you don't have an Octoclock to provide a reference PPS or 10MHz reference signal.
