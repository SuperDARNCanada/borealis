import json
import os


def ascii_encode_dict(data):
    ascii_encode = lambda x: x.encode('ascii')
    return dict(map(ascii_encode, pair) for pair in data.items())


class DataWriteOptions(object):
    """
    Parses the options from the config file that are relevant to data writing.

    """
    def __init__(self):
        super(DataWriteOptions, self).__init__()

        if not os.environ["BOREALISPATH"]:
            raise ValueError("BOREALISPATH env variable not set")
        config_path = os.environ["BOREALISPATH"] + "/config.ini"
        try:
            with open(config_path, 'r') as config_data:
                raw_config = json.load(config_data, object_hook=ascii_encode_dict)
        except IOError:
            errmsg = 'Cannot open config file at {0}'.format(config_path)
            raise IOError(errmsg)
        self._dsp_to_dw_identity = raw_config["dsp_to_dw_identity"]
        self._debug_file = raw_config["filter_outputs_debug_file"]
        self._data_directory = raw_config["data_directory"]
        self._site_id = raw_config["site_id"]
        self._rx_sample_rate = float(raw_config["rx_sample_rate"])
        self._max_usrp_dac_amplitude = float(raw_config["max_usrp_dac_amplitude"])
        self._pulse_ramp_time = float(raw_config["pulse_ramp_time"])
        self._tr_window_time = float(raw_config["tr_window_time"])
        self._output_sample_rate = float(raw_config["third_stage_sample_rate"])
        self._atten_window_time_start = float(raw_config["atten_window_time_start"])  # s
        self._atten_window_time_end = float(raw_config["atten_window_time_end"])  # s
        self._router_address = raw_config["router_address"]

    @property
    def dsp_to_dw_identity(self):
        """
        Gets the identity used for the dsp to data write socket.


        :returns: The identity to use for dsp/data_write socket.
        :rtype: str
        """
        return self._dsp_to_dw_identity

    @property
    def debug_file(self):
        """
        Gets the name of the file to output debug data to.

        :returns:   debug file name
        :rtype:     str
        """
        return self._debug_file

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
    def rx_sample_rate(self):
        """
        Gets the rx sample rate in samples per second.

        :return:    rx sample rate
        :rtype:     float
        """
        return self._rx_sample_rate

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
    def output_sample_rate(self):
        """
        Gets the sample rate of the output of the final filter stage in samples per second
        
        :return:    sample rate of the output of the final filter stage in samples per second
        :rtype:     float
        """
        return self._output_sample_rate

    @property
    def atten_window_time_start(self):
        """
        Gets the time before the RF pulse that the atten signal is active for in seconds.

        :return:    time before the RF pulse that the atten signal is active for in seconds.
        :rtype:     float
        """
        return self._atten_window_time_start

    @property
    def atten_window_time_end(self):
        """
        Gets the time after the RF pulse that the atten signal is active for in seconds.

        :return:    time after the RF pulse that the atten signal is active for in seconds.
        :rtype:     float
        """
        return self._atten_window_time_end

    @property
    def router_address(self):
        """
        Gets the socket address of the router that routes interprocess messages

        :return:    socket address of the router that routes interprocess messages
        :rtype:     str
        """
        return self._router_address

if __name__ == '__main__':
    DataWriteOptions()
