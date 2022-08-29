#!/usr/bin/python

# Copyright 2020 SuperDARN Canada
#
# signal_processing_options.py
# 2020-09-28
# options class for signal processing module

import json
import os

class SignalProcessingOptions(object):
    """
    Parses the options from the config file that are relevant to signal processing.

    """
    def __init__(self):
        super(SignalProcessingOptions, self).__init__()

        if not os.environ["BOREALISPATH"]:
            raise ValueError("BOREALISPATH env variable not set")
        config_path = os.environ["BOREALISPATH"] + "/config.ini"
        try:
            with open(config_path, 'r') as config_data:
                raw_config = json.load(config_data)
        except IOError:
            errmsg = 'Cannot open config file at {0}'.format(config_path)
            raise IOError(errmsg)

        self._router_address = raw_config["router_address"]
        self._dsp_radctrl_identity = raw_config["dsp_to_radctrl_identity"]
        self._dsp_driver_identity = raw_config["dsp_to_driver_identity"]
        self._dsp_exphan_identity = raw_config["dsp_to_exphan_identity"]
        self._dsp_dw_identity = raw_config["dsp_to_dw_identity"]
        self._dspbegin_brian_identity = raw_config["dspbegin_to_brian_identity"]
        self._dspend_brian_identity = raw_config["dspend_to_brian_identity"]
        self._radctrl_dsp_identity = raw_config["radctrl_to_dsp_identity"]
        self._driver_dsp_identity = raw_config["driver_to_dsp_identity"]
        self._brian_dspbegin_identity = raw_config["brian_to_dspbegin_identity"]
        self._brian_dspend_identity = raw_config["brian_to_dspend_identity"]
        self._exphan_dsp_identity = raw_config["exphan_to_dsp_identity"]
        self._dw_dsp_identity = raw_config["dw_to_dsp_identity"]
        self._ringbuffer_name = raw_config["ringbuffer_name"]
        self._main_antenna_count = int(raw_config["main_antenna_count"])
        self._intf_antenna_count = int(raw_config["interferometer_antenna_count"])
        self._main_antennas = []
        self._intf_antennas = []
        for n200 in raw_config["n200s"]:
            rx = bool(n200["rx"])
            tx = bool(n200["tx"])
            rx_int = bool(n200["rx_int"])
            if rx or tx:
                main_antenna_num = int(n200["main_antenna"])
                self._main_antennas.append(main_antenna_num)
            if rx_int:
                intf_antenna_num = int(n200["interferometer_antenna"])
                self._intf_antennas.append(intf_antenna_num)
        self._main_antennas.sort()
        self._intf_antennas.sort()


    @property
    def router_address(self):
        """
        Gets the socket address of the router that routes interprocess messages.

        :return:    socket address of the router that routes interprocess messages.
        :rtype:     str
        """
        return self._router_address

    @property
    def dsp_radctrl_identity(self):
        """
        Gets the socket identity for dsp to radar control.

        :returns:   dsp to radar control identity.
        :rtype:     str
        """

        return self._dsp_radctrl_identity

    @property
    def dsp_driver_identity(self):
        """
        Gets the socket identity for dsp to driver.

        :returns:   dsp to driver identity.
        :rtype:     str
        """
        return self._dsp_driver_identity

    @property
    def dsp_exphan_identity(self):
        """
        Gets the socket identity for dsp to exphan.

        :returns:   dsp to exphan identity.
        :rtype:     str
        """
        return self._dsp_exphan_identity

    @property
    def dsp_dw_identity(self):
        """
        Gets the socket identity for dsp to dw.

        :returns:   dsp to dw identity.
        :rtype:     str
        """
        return self._dsp_dw_identity

    @property
    def dspbegin_brian_identity(self):
        """
        Gets the socket identity for dspbegin to brian.

        :returns:   dspbegin to brian identity.
        :rtype:     str
        """
        return self._dspbegin_brian_identity

    @property
    def dspend_brian_identity(self):
        """
        Gets the socket identity for dspend to brian.

        :returns:   dspend to brian identity.
        :rtype:     str
        """
        return self._dspend_brian_identity

    @property
    def radctrl_dsp_identity(self):
        """
        Gets the socket identity for radctrl to dsp.

        :returns:   radctrl to dsp identity.
        :rtype:     str
        """
        return self._radctrl_dsp_identity

    @property
    def driver_dsp_identity(self):
        """
        Gets the socket identity for driver to dsp.

        :returns:   driver to dsp identity.
        :rtype:     str
        """
        return self._driver_dsp_identity

    @property
    def brian_dspbegin_identity(self):
        """
        Gets the socket identity for brian to dspbegin.

        :returns:   brian to dspbegin identity.
        :rtype:     str
        """
        return self._brian_dspbegin_identity

    @property
    def brian_dspend_identity(self):
        """
        Gets the socket identity for brian to dspend.

        :returns:   Brian to dspend identity.
        :rtype:     str
        """
        return self._brian_dspend_identity

    @property
    def exphan_dsp_identity(self):
        """
        Gets the socket identity for exphan to dsp.

        :returns:   Exphan to dsp identity.
        :rtype:     str
        """
        return self._exphan_dsp_identity

    @property
    def dw_dsp_identity(self):
        """
        Gets the socket identity for dw to dsp.

        :returns:   dw to dsp identity.
        :rtype:     str
        """
        return self._dw_dsp_identity

    @property
    def ringbuffer_name(self):
        """
        Gets the shared memory ringbuffer name.

        :returns:   The ringbuffer name.
        :rtype:     str
        """
        return self._ringbuffer_name

    @property
    def main_antenna_count(self):
        """
        Gets the main antenna count.

        :returns:   Number of main antennas.
        :rtype:     int
        """
        return self._main_antenna_count

    @property
    def intf_antenna_count(self):
        """
        Gets the intf antenna count.

        :returns:   Number of intf antennas.
        :rtype:     int
        """
        return self._intf_antenna_count


if __name__ == '__main__':
    SignalProcessingOptions()
