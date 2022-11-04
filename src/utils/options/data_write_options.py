#!/usr/bin/python

# Copyright 2017 SuperDARN Canada
#
# data_write_options.py
# 2018-05-14
# options class for data write module
# TODO: Get experiment details from somewhere to write metadata out to files (freq, cpid, etc..)

import json
import os

class DataWriteOptions(object):
    """
    Parses the options from the config file that are relevant to data writing.

    """
    def __init__(self):
        super(DataWriteOptions, self).__init__()

        if not os.environ["BOREALISPATH"]:
            raise ValueError("BOREALISPATH env variable not set")
        if not os.environ["RADAR_CODE"]:
            raise ValueError("RADAR_CODE env variable not set")
        config_path = f'{os.environ["BOREALISPATH"]}/config/{os.environ["RADAR_CODE"]}/{os.environ["RADAR_CODE"]}_config.ini'
        try:
            with open(config_path, 'r') as config_data:
                raw_config = json.load(config_data)
        except IOError:
            errmsg = 'Cannot open config file at {0}'.format(config_path)
            raise IOError(errmsg)

        self._rt_to_dw_identity = raw_config["rt_to_dw_identity"]
        self._dw_to_rt_identity = raw_config["dw_to_rt_identity"]
        self._dsp_to_dw_identity = raw_config["dsp_to_dw_identity"]
        self._dw_to_dsp_identity = raw_config["dw_to_dsp_identity"]
        self._radctrl_to_dw_identity = raw_config["radctrl_to_dw_identity"]
        self._dw_to_radctrl_identity = raw_config["dw_to_radctrl_identity"]
        self._data_directory = raw_config["data_directory"]
        self._site_id = raw_config["site_id"]
        self._max_usrp_dac_amplitude = float(raw_config["max_usrp_dac_amplitude"])
        self._pulse_ramp_time = float(raw_config["pulse_ramp_time"])
        self._tr_window_time = float(raw_config["tr_window_time"])
        self._router_address = raw_config["router_address"]
        self._main_antenna_count = int(raw_config["main_antenna_count"])
        self._intf_antenna_count = int(raw_config["interferometer_antenna_count"])

        # Parse N200 array and calculate main and intf antennas operating
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
    def rt_to_dw_identity(self):
        """
        Gets the identity used for the realtime to datawrite identity.

        Returns:
            String: The identity to use for realtime/datawrite socket.
        """
        return self._rt_to_dw_identity

    @property
    def dw_to_rt_identity(self):
        """
        Gets the identity used for the datawrite to realtime identity.

        Returns:
            String: The identity to use for the datawrite/realtime socket.
        """
        return self._dw_to_rt_identity

    @property
    def dsp_to_dw_identity(self):
        """
        Gets the identity used for the dsp to data write socket.

        :returns: The identity to use for dsp/data_write socket.
        :rtype: str
        """
        return self._dsp_to_dw_identity

    @property
    def dw_to_dsp_identity(self):
        """
        Gets the identity used for the data write to dsp socket.

        :returns: The identity to use for data_write/dsp socket.
        :rtype: str
        """
        return self._dw_to_dsp_identity

    @property
    def radctrl_to_dw_identity(self):
        """Gets the identity used for the radar control to data write socket.

        Returns: The identity used for radar control/data write socket.
        TYPE: str
        """
        return self._radctrl_to_dw_identity

    @property
    def dw_to_radctrl_identity(self):
        """Gets the identity used for the data write to radar control socket.

        Returns: The identity used for data write/radar control socket.
        TYPE: str
        """
        return self._dw_to_radctrl_identity


    # @property
    # def debug_file(self):
    #     """
    #     Gets the name of the file to output debug data to.

    #     :returns:   debug file name
    #     :rtype:     str
    #     """
    #     return self._debug_file

    @property
    def data_directory(self):
        """
        Gets the location of the directory to place data files in.

        :returns:  Data directory location
        :rtype:    str
        """
        return self._data_directory

    @property
    def site_id(self):
        """
        Gets the 3 letter radar code of this radar.

        :return:    3 letter radar code
        :rtype:     str
        """
        return self._site_id

    @property
    def max_usrp_dac_amplitude(self):
        """
        Gets the maximum usrp dac amplitude, which is a value usually between -1 and 1

        :return:    maximum dac amplitude of USRP units
        :rtype:     float
        """
        return self._max_usrp_dac_amplitude

    @property
    def pulse_ramp_time(self):
        """
        Gets the ramp-up and ramp-down time of the RF pulses in seconds

        :return:    ramp-up/ramp-down time of the RF pulse in seconds.
        :rtype:     float
        """
        return self._pulse_ramp_time

    @property
    def tr_window_time(self):
        """
        Gets the time before and after the RF pulse that the TR signal is active for in seconds.

        :return:    time before and after the RF pulse that TR signal is active for in seconds
        :rtype:     float
        """
        return self._tr_window_time

    @property
    def router_address(self):
        """
        Gets the socket address of the router that routes interprocess messages.

        :return:    socket address of the router that routes interprocess messages.
        :rtype:     str
        """
        return self._router_address

    @property
    def main_antenna_count(self):
        """
        Gets the number of main array antennas.

        :return:    number of main antennas.
        :rtype:     int
        """

        return self._main_antenna_count

    @property
    def intf_antenna_count(self):
        """
        Gets the number of interferometer array antennas.

        :return:    number of interferometer antennas.
        :rtype:     int
        """

        return self._intf_antenna_count

    @property
    def main_antennas(self):
        """
        Gets the index of antennas in the main array corresponding to the transceiver channels.

        :return:    indices of transceiver channels mapped to antennas in main array.
        :rtype:     list[int]
        """

        return self._main_antennas

    @property
    def intf_antennas(self):
        """
        Gets the index of antennas in the interferometer array corresponding to the receiver channels.

        :return:    indices of receiver channels mapped to antennas in interferometer array.
        :rtype:     list[int]
        """

        return self._intf_antennas

if __name__ == '__main__':
    DataWriteOptions()
