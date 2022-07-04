*****************
Config Parameters
*****************
+--------------------------------+---------------------------+--------------------------------------+
| Config field                   | Example entry             | Description                          |
+================================+===========================+======================================+
| site_id                        | sas                       | 3-letter standard ID of the radar    |
+--------------------------------+---------------------------+--------------------------------------+
| gps_octoclock_addr             | addr=192.168.10.131       | IP address of the GPS Octoclock      |
+--------------------------------+---------------------------+--------------------------------------+
| device_options                 | recv_frame_size=4000      | UHD USRP device arguments.           |
+--------------------------------+---------------------------+--------------------------------------+
| main_antenna_count             | 16                        | Number of main array antennas (TX/RX)|
+--------------------------------+---------------------------+--------------------------------------+
| main_antennas                  | 0,1,2,3,4,5,6,7,8,9,      | Mapping of main rx/tx channels to    |
|                                | 10,11,12,13,14,15         | antennas in the main array           |
+--------------------------------+---------------------------+--------------------------------------+
| interferometer_antenna_count   | 4                         | Number of interferometer antennas    |
+--------------------------------+---------------------------+--------------------------------------+
| interferometer_antennas        | 1,3,5,7                   | Mapping of intf rx channels to       |
|                                |                           | antennas in the interferometer array |
+--------------------------------+---------------------------+--------------------------------------+
| n200s                          | | {                       | List of all N200s, both active and   |
|                                | | addr : "192.168.10.100" | not. Order of N200s is specified by  |
|                                | | isActivated : true      | main_antenna, so N200s can be listed |
|                                | | main_antenna : "0"      | out of order and more than the number|
|                                | | int_antenna : ""        | of antennas can be stored here (extra|
|                                | | }, {                    | ones must have isActivated = false). |
|                                | | addr : "192.168.10.101" |                                      |
|                                | | isActivated : true      |                                      |
|                                | | main_antenna : "1"      |                                      |
|                                | | int_antenna : "0"       |                                      |
|                                | | }, ...                  |                                      |
+--------------------------------+---------------------------+--------------------------------------+
| addr (n200s)                   | 192.168.10.100            | IP address of the specified N200     |
+--------------------------------+---------------------------+--------------------------------------+
| isActivated (n200s)            | true                      | All activated N200s must have this   |
|                                |                           | set to true, and false otherwise     |
+--------------------------------+---------------------------+--------------------------------------+
| main_antenna (n200s)           | 2                         | Physical antenna connected to this   |
|                                |                           | N200. N200s are ordered according to |
|                                |                           | this value.                          |
+--------------------------------+---------------------------+--------------------------------------+
| int_antenna (n200s)            | 0                         | Interferometer antenna connected to  |
|                                |                           | this N200. All antennas not connected|
|                                |                           | must be set to empty string ("")     |
+--------------------------------+---------------------------+--------------------------------------+
| main_antenna_spacing           | 15.24                     | Distance between antennas (m).       |
+--------------------------------+---------------------------+--------------------------------------+
| interferometer_antenna_spacing | 15.24                     | Distance between antennas (m).       |
+--------------------------------+---------------------------+--------------------------------------+
| min_freq                       | 8.00E+06                  | Minimum frequency we can run (Hz).   |
+--------------------------------+---------------------------+--------------------------------------+
| max_freq                       | 20.00E+06                 | Maximum frequency we can run (Hz).   |
+--------------------------------+---------------------------+--------------------------------------+
| minimum_pulse_length           | 100                       | Minimum pulse length (us) dependent  |
|                                |                           | upon AGC feedback sample and hold.   |
+--------------------------------+---------------------------+--------------------------------------+
| minimum_mpinc_length           | 1                         | Minimum length of multi-pulse        |
|                                |                           | increment (us).                      |
+--------------------------------+---------------------------+--------------------------------------+
| minimum_pulse_separation       | 125                       | The minimum separation (us) before   |
|                                |                           | experiment treats it as a single     |
|                                |                           | pulse (transmitting zeroes and not   |
|                                |                           | receiving between the pulses. 125 us |
|                                |                           | is approx two TX/RX times.           |
+--------------------------------+---------------------------+--------------------------------------+
| tx_subdev                      | A:A                       | UHD daughterboard string which       |
|                                |                           | defines how to configure ports. Refer|
|                                |                           | to UHD subdev docs.                  |
+--------------------------------+---------------------------+--------------------------------------+
| max_tx_sample_rate             | 5.00E+06                  | Maximum wideband TX rate each device |
|                                |                           | can run in the system.               |
+--------------------------------+---------------------------+--------------------------------------+
| main_rx_subdev                 | A:A A:B                   | UHD daughterboard string which       |
|                                |                           | defines how to configure ports. Refer|
|                                |                           | to UHD subdev docs.                  |
+--------------------------------+---------------------------+--------------------------------------+
| interferometer_rx_subdev       | A:A A:B                   | UHD daughterboard string which       |
|                                |                           | defines how to configure ports. Refer|
|                                |                           | to UHD subdev docs.                  |
+--------------------------------+---------------------------+--------------------------------------+
| max_rx_sample_rate             | 5.00E+06                  | Maximum wideband RX rate each        |
|                                |                           | device can run in the system.        |
+--------------------------------+---------------------------+--------------------------------------+
| pps                            | external                  | The PPS source for the system        |
|                                |                           | (internal, external, none).          |
+--------------------------------+---------------------------+--------------------------------------+
| ref                            | external                  | The 10 MHz reference source          |
|                                |                           | (internal, external).                |
+--------------------------------+---------------------------+--------------------------------------+
| overthewire                    | sc16                      | Data type for samples the USRP       |
|                                |                           | operates with. Refer to UHD docs for |
|                                |                           | data types.                          |
+--------------------------------+---------------------------+--------------------------------------+
| cpu                            | fc32                      | Data type of samples that UHD uses   |
|                                |                           | on host CPU. Refer to UHD docs for   |
|                                |                           | data types.                          |
+--------------------------------+---------------------------+--------------------------------------+
| gpio_bank_high                 | RXA                       | The daughterboard pin bank to use for|
|                                |                           | active-high TR and I/O signals.      |
+--------------------------------+---------------------------+--------------------------------------+
| gpio_bank_low                  | TXA                       | The daughterboard pin bank to use for|
|                                |                           | active-low TR and I/O signals.       |
+--------------------------------+---------------------------+--------------------------------------+
| atr_rx                         | 0x0006                    | The pin mask for the RX only signal. |
+--------------------------------+---------------------------+--------------------------------------+
| atr_tx                         | 0x0018                    | The pin mask for the TX only signal. |
+--------------------------------+---------------------------+--------------------------------------+
| atr_xx                         | 0x0060                    | The pin mask for the full duplex     |
|                                |                           | signal (TR).                         |
+--------------------------------+---------------------------+--------------------------------------+
| atr_0x                         | 0x0180                    | The pin mask for the idle signal.    |
+--------------------------------+---------------------------+--------------------------------------+
| tst_md                         | 0x0600                    | The pin mask for the test mode signal|
+--------------------------------+---------------------------+--------------------------------------+
| lo_pwr                         | 0x1800                    | The pin mask for the low power signal|
+--------------------------------+---------------------------+--------------------------------------+
| agc_st                         | 0x6000                    | The pin mask for the AGC signal.     |
+--------------------------------+---------------------------+--------------------------------------+
| max_usrp_dac_amplitude         | 0.99                      | The amplitude of highest allowed USRP|
|                                |                           | TX sample (V).                       |
+--------------------------------+---------------------------+--------------------------------------+
| pulse_ramp_time                | 1.00E-05                  | The linear ramp time for the         |
|                                |                           | pulse (s)                            |
+--------------------------------+---------------------------+--------------------------------------+
| tr_window_time                 | 6.00E-05                  | How much windowing on either side of |
|                                |                           | pulse is needed for TR signal (s).   |
+--------------------------------+---------------------------+--------------------------------------+
| agc_signal_read_delay          | 0                         | Hardware dependent delay after seq   |
|                                |                           | is finished for reading              |
|                                |                           | of AGC and low power signals (s)     |
+--------------------------------+---------------------------+--------------------------------------+
| usrp_master_clock_rate         | 1.00E+08                  | Clock rate of the USRP master        |
|                                |                           | clock (Sps).                         |
+--------------------------------+---------------------------+--------------------------------------+
| max_output_sample_rate         | 1.00E+05                  | Maximum rate allowed after           |
|                                |                           | downsampling (Sps)                   |
+--------------------------------+---------------------------+--------------------------------------+
| max_number_of_filter_taps      | 2048                      | The maximum total number of filter   |
| _per_stage                     |                           | taps for all frequencies combined.   |
|                                |                           | This is a GPU limitation.            |
+--------------------------------+---------------------------+--------------------------------------+
| router_address                 | tcp://127.0.0.1:6969      | The protocol/IP/port used for the ZMQ|
|                                |                           | router in Brian.                     |
+--------------------------------+---------------------------+--------------------------------------+
| radctrl_to_exphan_identity     | RADCTRL_EXPHAN_IDEN       | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| radctrl_to_dsp_identity        | RADCTRL_DSP_IDEN          | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| radctrl_to_driver_identity     | RADCTRL_DRIVER_IDEN       | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| radctrl_to_brian_identity      | RADCTRL_BRIAN_IDEN        | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| radctrl_to_dw_identity         | RADCTRL_DW_IDEN           | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| driver_to_radctrl_identity     | DRIVER_RADCTRL_IDEN       | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| driver_to_dsp_identity         | DRIVER_DSP_IDEN           | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| driver_to_brian_identity       | DRIVER_BRIAN_IDEN         | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| exphan_to_radctrl_identity     | EXPHAN_RADCTRL_IDEN       | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| exphan_to_dsp_identity         | EXPHAN_DSP_IDEN           | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| dsp_to_radctrl_identity        | DSP_RADCTRL_IDEN          | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| dsp_to_driver_identity         | DSP_DRIVER_IDEN           | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| dsp_to_exphan_identity         | DSP_EXPHAN_IDEN           | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| dsp_to_dw_identity             | DSP_DW_IDEN               | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| dspbegin_to_brian_identity     | DSPBEGIN_BRIAN_IDEN       | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| dspend_to_brian_identity       | DSPEND_BRIAN_IDEN         | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| dw_to_dsp_identity             | DW_DSP_IDEN               | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| dw_to_radctrl_identity         | DW_RADCTRL_IDEN           | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| brian_to_radctrl_identity      | BRIAN_RADCTRL_IDEN        | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| brian_to_driver_identity       | BRIAN_DRIVER_IDEN         | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| brian_to_dspbegin_identity     | BRIAN_DSPBEGIN_IDEN       | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| brian_to_dspend_identity       | BRIAN_DSPEND_IDEN         | ZMQ named socket identity.           |
+--------------------------------+---------------------------+--------------------------------------+
| ringbuffer_name                | data_ringbuffer           | Shared memory name for ringbuffer.   |
+--------------------------------+---------------------------+--------------------------------------+
| ringbuffer_size_bytes          | 200.00E+06                | Size in bytes to allocate for each   |
|                                |                           | ringbuffer.                          |
+--------------------------------+---------------------------+--------------------------------------+
| data_directory                 | /data/borealis_data       | Location of output data files.       |
+--------------------------------+---------------------------+--------------------------------------+

**********************
Example configurations
**********************
There are several instances when you'll need to modify this file for correct operation.

#. One of your main array antennas is not working properly (broken coax, blown lightning arrestor, etc)

    The main antenna channel mapping associated with the bad antenna should be removed from
    main_antenna_usrp_rx_channels. This will disable the N200s from collecting samples from that antenna.

#. One of your interferometer array antennas is not working properly (broken coax, blown lightning arrestor, etc)

    The interferometer antenna channel mapping associated with the bad antenna should be removed from
    interferometer_antenna_usrp_rx_channels. This will disable the N200s from collecting samples from that antenna.

#. One of your transmitter's transmit paths is not working, but the receive path is still working properly

    The channel mapping associated with the bad transmitter should be removed from the main_antenna_usrp_tx_channels.
    This will disable transmit on the bad transmit path.

#. One of your transmitter's receive paths is not working, but the transmit path is still working properly

    The main antenna channel mapping associated with the bad transmitter should be removed from
    main_antenna_usrp_rx_channels. This will disable the N200s from collecting samples from that receive
    path.

#. One of your transmitters is not working at all

    The main antenna channel mapping associated with the bad transmitter should be removed from
    main_antenna_usrp_rx_channels. This will disable the N200s from collecting samples from that receive
    path. The channel mapping associated with the bad transmitter should be removed from the
    main_antenna_usrp_tx_channels. This will disable transmit on the bad transmit path.

#. One of your N200s is not working properly and you've inserted the spare N200

    In this instance, since you still have the same number of antennas as well as transmit and receive channels,
    you simply need to change the IP adress of the N200 you replaced. This is done in the `devices` config option.
    An example: if N200 with IP address 192.168.10.104 dies, and is replaced with the spare (ip address 192.168.10.116),
    simply replace `addr4=192.168.10.104` with `addr4=192.168.10.116`.

#. One of your N200s is not working properly but you're located remotely and cannot insert the spare N200

    This particular N200 will have to be removed from the config file. The transmitter and receive
    paths that this N200 is connected to will be disabled. The address needs to be removed from the
    list of addresses and the address numbering needs to be adjusted. The main and interferometer
    channel mappings will be need to be adjusted. The main and interferometer antenna counts need to
    be adjusted. When this N200 is replaced, these options will have to be restored.

#. You have a non-standard array

    One example of a non-standard array would be a different number of interferometer antennas than four.
    If your interferometer array has only two antennas you'll need to modify the following:

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
