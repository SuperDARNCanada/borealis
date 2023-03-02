#!/usr/bin/python

"""
    realtime options
    ~~~~~~~~~~~~~~~~

    To load the config options to be used by realtime
    Config data comes from the config.ini file

    :copyright: 2019 SuperDARN Canada
"""

from ..general import load_config


class RealtimeOptions(object):
    """
    Parses the options from the config file that are relevant to realtime.

    """
    def __init__(self):
        super(RealtimeOptions, self).__init__()

        # Gather the borealis configuration information
        raw_config = load_config()

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
