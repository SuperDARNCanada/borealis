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
        with open(config_path,'r') as f:
            raw_config = json.load(f,object_hook=ascii_encode_dict)


        self._rx_dsp_to_data_write_address = raw_config["rx_dsp_to_data_write_address"]
        self._debug_file = raw_config["filter_outputs_debug_file"]

    @property
    def rx_dsp_to_data_write_address(self):
        """
        Gets the address used for the rx dsp to data write socket.


        :returns: The address to bind to for dsp/data_write socket.
        :rtype: str
        """
        return self._rx_dsp_to_data_write_address

    @property
    def debug_file(self):
        """
        Gets the name of the file to output debug data to.

        :returns:   debug file name
        :rtype:     str
        """
        return self._debug_file



if __name__ == '__main__':
    DataWriteOptions()