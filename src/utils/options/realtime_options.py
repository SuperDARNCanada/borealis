#!/usr/bin/python

"""
    realtime options
    ~~~~~~~~~~~~~~~~

    To load the config options to be used by realtime
    Config data comes from the config.ini file

    :copyright: 2019 SuperDARN Canada
"""

import json
import os

class RealtimeOptions(object):
    """
    Parses the options from the config file that are relevant to realtime.

    """
    def __init__(self):
        super(RealtimeOptions, self).__init__()

        if not os.environ["BOREALISPATH"]:
            raise ValueError("BOREALISPATH env variable not set")
        if not os.environ['RADAR_CODE']:
            raise ValueError('RADAR_CODE env variable not set')
        config_path = f'{os.environ["BOREALISPATH"]}/config/{os.environ["RADAR_CODE"]}/{os.environ["RADAR_CODE"]}_config.ini'
        try:
            with open(config_path, 'r') as config_data:
                raw_config = json.load(config_data)
        except IOError:
            errmsg = f'Cannot open config file at {config_path}'
            raise IOError(errmsg)

        self._rt_to_dw_identity = raw_config["rt_to_dw_identity"]
        self._dw_to_rt_identity = raw_config["dw_to_rt_identity"]
        self._rt_address = raw_config["realtime_address"]
        self._router_address = raw_config["router_address"]


    @property
    def rt_to_dw_identity(self):
        """
        Gets the identity used for the realtime to datawrite identity.

        :returns:   The identity to use for realtime/datawrite socket.
        :rtype:     str
        """
        return self._rt_to_dw_identity

    @property
    def dw_to_rt_identity(self):
        """
        Gets the identity used for the datawrite to realtime identity.

        :returns:   The identity to use for the datawrite/realtime socket.
        :rtype:     str
        """
        return self._dw_to_rt_identity

    @property
    def rt_address(self):
        """
        Gets the address used to bind on for realtime applications.

        :returns:   The address used to bind on.
        :rtype:     str
        """
        return self._rt_address

    @property
    def router_address(self):
        """
        Gets the socket address of the router that routes interprocess messages.

        :returns:   socket address of the router that routes interprocess messages.
        :rtype:     str
        """
        return self._router_address
