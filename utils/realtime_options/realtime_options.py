#!/usr/bin/python

# Copyright 2019 SuperDARN Canada
#
# realtime_options.py
# options class for realtime module

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
        config_path = os.environ["BOREALISPATH"] + "/config.ini"
        try:
            with open(config_path, 'r') as config_data:
                raw_config = json.load(config_data)
        except IOError:
            errmsg = 'Cannot open config file at {0}'.format(config_path)
            raise IOError(errmsg)

        self._rt_to_dw_identity = raw_config["rt_to_dw_identity"]
        self._dw_to_rt_identity = raw_config["dw_to_rt_identity"]
        self._rt_address = raw_config["realtime_address"]
        self._router_address = raw_config["router_address"]


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
    def rt_address(self):
        """
        Gets the address used to bind on for realtime applications.

        Returns:
            String: The address used to bind on.
        """
        return self._rt_address

    @property
    def router_address(self):
        """
        Gets the socket address of the router that routes interprocess messages.

        :return:    socket address of the router that routes interprocess messages.
        :rtype:     str
        """
        return self._router_address
