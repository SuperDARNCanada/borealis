.. _config-options:

=============
Configuration
=============

Below is an example configuration file. You can copy this file (e.g. to a file called ``template_config``) and strip
the comments as a starter for your own config file, using ``sed``::

    sed 's/#.*//' template_config | sed -e 's/[[:space:]]*$//g' > my_config.ini

..  literalinclude:: ../../tests/config_files/base_config.ini
    :caption: Example Config File

---------------------
Testing a config file
---------------------
A test script is available at ``tests/config_files/config_testing.py`` for verifying the fields of your configuration
file. This script is detailed further in :ref:`Config Testing <Config Testing>`.

----------------------
Example configurations
----------------------
There are several instances when you'll need to modify this file for correct operation.

#. One of your main array antennas is not working properly (broken coax, blown lightning arrestor,
   etc)

    The N200(s) with RX and TX channels connected to that antenna should have those channels set to ``""``.
    This will disable transmission and reception on the antenna, while preserving the correct phasing
    for beamforming on other channels. E.g.::

        "n200s" : [                         # List of N200 configuration options
            {
                "addr" : "192.168.10.100",  # IP address of the USRP previously connected to "m0"
                "rx_channel_0" : "",        # RX channel 0 no longer connected to "m0"
                "rx_channel_1" : "i0",      # No change
                "tx_channel_0" : ""         # TX channel 0 no longer connected to "m0"
            },
            ...
        ],

#. One of your interferometer array antennas is not working properly (broken coax, blown lightning
   arrestor, etc)

    The N200 with an RX channel connected to that antenna should have that channel set to ``""``.
    This will disable reception from the antenna, while preserving the correct phasing for beamforming
    data from other channels. E.g.::

        "n200s" : [                         # List of N200 configuration options
            {
                "addr" : "192.168.10.100",  # IP address of the USRP previously connected to "i0"
                "rx_channel_0" : "m0",      # No change
                "rx_channel_1" : "",        # RX channel 1 no longer connected to "i0"
                "tx_channel_0" : "m0"       # No change
            },
            ...
        ],

#. One of your transmitter's transmit paths is not working, but the receive path is still working
   properly

    The ``tx_channel_0`` field for the associated N200 should be set to ``""``. This will disable the transmission
    channel on the bad transmit path. E.g.::

        "n200s" : [                         # List of N200 configuration options
            {
                "addr" : "192.168.10.100",  # IP address of the USRP
                "rx_channel_0" : "m0",      # No change
                "rx_channel_1" : "i0",      # No change
                "tx_channel_0" : ""         # TX channel 0 no longer connected to "m0"
            },
            ...
        ],

#. One of your transmitter's receive paths is not working, but the transmit path is still working
   properly

    The ``rx_channel_#`` flag for the associated N200 should be set to ``""``. This will disable the receive
    channel on the bad receive path. E.g.::

        "n200s" : [                         # List of N200 configuration options
            {
                "addr" : "192.168.10.100",  # IP address of the affected USRP
                "rx_channel_0" : "",        # RX channel 0 no longer connected to "m0"
                "rx_channel_1" : "i0",      # No change
                "tx_channel_0" : "m0"       # No change
            },
            ...
        ],

#. One of your transmitters is not working at all

    Ensure that no N200s have a channel set to use that corresponding antenna. For example, if the transmitter
    for antenna 7 of the main array is broken, make sure no ``rx_channel_#`` or ``tx_channel_0`` fields are set
    to ``"m7"``. E.g.::

        "n200s" : [                         # List of N200 configuration options
            ...,
            {
                "addr" : "192.168.10.107",  # IP address of the USRP previously connected to "m7"
                "rx_channel_0" : "",        # RX channel 0 no longer connected to "m7"
                "rx_channel_1" : "",        # No change
                "tx_channel_0" : ""         # TX channel 0 no longer connected to "m7"
            },
            ...
        ],

#. One of your N200s is not working properly and you've inserted the spare N200

    Add an entry for the replacement N200, and copy ``rx_channel_0``, ``rx_channel_1``, and ``tx_channel_0``
    fields from the broken N200. Make sure that the cables are transferred over, and verify that the cabling
    matches with the antennas specified for each channel. The configuration for the broken N200 can be set to ``""``
    for each channel. If all channels are set to ``""``, the N200 is ignored. E.g.::

        "n200s" : [                         # List of N200 configuration options
            {
                "addr" : "192.168.10.100",  # IP address of the broken USRP
                "rx_channel_0" : "",        # Remove connection
                "rx_channel_1" : "",        # Remove connection
                "tx_channel_0" : ""         # Remove connection
            },
            ...,
            {
                "addr" : "192.168.10.116",  # IP address of the spare USRP
                "rx_channel_0" : "m0",      # Connect to antenna previously used by broken USRP
                "rx_channel_1" : "i0",      # Connect to antenna previously used by broken USRP
                "tx_channel_0" : "m0"       # Connect to antenna previously used by broken USRP
            },
        ],

#. One of your N200s is not working properly but you're located remotely and cannot insert the spare
   N200

    This particular N200 will have to be deactivated. To do this, set all channel fields to ``""``. E.g.::

        "n200s" : [                         # List of N200 configuration options
            ...,
            {
                "addr" : "192.168.10.107",  # IP address of the broken USRP
                "rx_channel_0" : "",        # No connection
                "rx_channel_1" : "",        # No connection
                "tx_channel_0" : ""         # No connection
            },
            ...
        ],


#. You have a non-standard array

    One example of a non-standard array would be a different number of interferometer antennas than
    four. To implement this, modify the individual N200 entries to specify which N200s are connected
    to interferometer antennas. Additionally, set the main and interferometer antenna count
    parameters to the number of physical antennas in each array. E.g.::

        "antennas" : {
            "main_locations": {
                "0" : [-114.3, 0.0, 0.0],
                "1" : [-99.06, 0.0, 0.0],
                "2" : [-83.82, 0.0, 0.0],
                "3" : [-68.58, 0.0, 0.0],
                "4" : [-53.34, 0.0, 0.0],
                "5" : [-38.10, 0.0, 0.0],
                "6" : [-22.86, 0.0, 0.0],
                "7" : [-7.62, 0.0, 0.0],
                "8" : [7.62, 0.0, 0.0],
                "9" : [22.86, 0.0, 0.0],
                "10" : [38.10, 0.0, 0.0],
                "11" : [53.34, 0.0, 0.0],
                "12" : [68.58, 0.0, 0.0],
                "13" : [83.82, 0.0, 0.0],
                "14" : [99.06, 0.0, 0.0],
                "15" : [114.3, 0.0, 0.0]
            },
            "intf_locations": {
                "0" : [-22.86, -100.0, 0.0],
                "1" : [-7.62, -100.0, 0.0],
                "2" : [7.62, -100.0, 0.0]     # Last antenna removed
            },
            "main_antenna_count" : "16",
            "intf_antenna_count" : "3",
            "main_antenna_spacing": "15.24",
            "intf_antenna_spacing": "15.24",
            "standard_positions": true
        },

#. You want to change the location of ATR signals on the daughterboards

    This can be done by changing the values of the following config parameters: atr_rx, atr_tx,
    atr_xx, atr_0x, tst_md, lo_pwr, agc_st. The value ``atr_rx = 0x0006`` means that the ATR_RX
    signal will appear on the pins 1 and 2 (referenced from 0). I.e. every bit that is a '1' in this
    hex value indicates which pin the signal will appear on. The standard pin mappings are::

        "atr_rx" : "0x0006",  # Pin mask for the RX-Only signal
        "atr_tx" : "0x0018",  # Pin mask for the TX-Only signal
        "atr_xx" : "0x0060",  # Pin mask for the Full Duplex (T/R) signal
        "atr_0x" : "0x0180",  # Pin mask for the Idle signal
        "lo_pwr" : "0x0600",  # Pin mask for the Low Power signal
        "agc_st" : "0x1800",  # Pin mask for the AGC Status signal
        "tst_md" : "0x6000",  # Pin mask for the Test Mode signal

#. You want to change the polarity of the ATR signals on the daughterboards

    This can be done by swapping the values of the two config parameters: ``gpio_bank_high`` and
    ``gpio_bank_low``. The default is for active-high signals to be on the LFRX daughterboard. This
    is done by setting ``gpio_bank_high`` to ``RXA``. The same signals, but active-low, are by
    default located on the LFTX daughterboard. E.g.::

        "gpio_bank_high" : "RXA",  # Daughterboard pin bank to use for active-high TR and I/O signals
        "gpio_bank_low" : "TXA",   # Daughterboard pin bank to use for active-low TR and I/O signals

#. You would like to make a test-system with only one N200 and don't have any Octoclocks

    This can be done by changing the following parameters:

    #. ``n200s`` - Set ``tx_channel_0``, ``rx_channel_0``, and ``rx_channel_1`` fields for only one N200. All
       others should have their channels set to ``""``.

    #. ``pps`` and ``ref`` - These should both be set to ``internal``, as you don't have an
       Octoclock to provide a reference PPS or 10MHz reference signal. E.g.::

          "pps" : "internal",  # Pulse-per-second reference source is USRP clock
          "ref" : "internal",  # 10 MHz reference source is USRP clock
