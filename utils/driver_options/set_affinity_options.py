#!/usr/bin/python

# Copyright 2018 SuperDARN Canada
#
# set_affinity_options.py
# 2018-05-14
# options class for set_affinity module

import json
import os


def ascii_encode_dict(data):
    ascii_encode = lambda x: x.encode('ascii')
    return dict(map(ascii_encode, pair) for pair in data.items())


class SetAffinityOptions(object):
    """
    Parses the options from the config file that are relevant to setting driver thread affinity.

    """
    def __init__(self):
        super(SetAffinityOptions, self).__init__()

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
        self._driver_to_mainaffinity_identity = raw_config["driver_to_mainaffinity_identity"]
        self._driver_to_txaffinity_identity = raw_config["driver_to_txaffinity_identity"]
        self._driver_to_rxaffinity_identity = raw_config["driver_to_rxaffinity_identity"]
        self._mainaffinity_to_driver_identity = raw_config["mainaffinity_to_driver_identity"]
        self._txaffinity_to_driver_identity = raw_config["txaffinity_to_driver_identity"]
        self._rxaffinity_to_driver_identity = raw_config["rxaffinity_to_driver_identity"]
        self._device_str = raw_config["device_options"]

        # Parse N200 array and construct devices string
        devices_map = {}
        for n200 in raw_config["n200s"]:
            rx = bool(n200["rx"])
            tx = bool(n200["tx"])
            rx_int = bool(n200["rx_int"])

            if rx or tx or rx_int:
                device_num = int(n200["main_antenna"])
                devices_map[device_num] = n200["addr"]

        addr_idx = 0
        for device_num in sorted(devices_map.keys()):
            self._device_str = self._device_str + ",addr" + str(addr_idx) + "=" + devices_map[device_num]
            addr_idx += 1


    @property
    def device_str(self):
        """Gets the device string from config file.

        :returns: Gets the device string from config file.
        :rtype: String
        """
        return self._device_str


    @property
    def driver_to_mainaffinity_identity(self):
        """Gets the socket name for driver to main thread affinity

        :returns: Gets the socket name for driver to main thread affinity
        :rtype: String
        """
        return self._driver_to_mainaffinity_identity

    @property
    def driver_to_txaffinity_identity(self):
        """Gets the socket name for driver to tx thread affinity.

        :returns: Gets the socket name for driver to tx thread affinity.
        :rtype: String
        """
        return self._driver_to_txaffinity_identity

    @property
    def driver_to_rxaffinity_identity(self):
        """Gets the socket name for the driver to rx thread affinity.

        :returns: Gets the socket name for the driver to rx thread affinity.
        :rtype: String
        """
        return self._driver_to_rxaffinity_identity

    @property
    def mainaffinity_to_driver_identity(self):
        """Gets the socket name for the main affinity to driver.

        :returns: Gets the socket name for the main affinity to driver.
        :rtype:   String
        """
        return self._mainaffinity_to_driver_identity

    @property
    def txaffinity_to_driver_identity(self):
        """Gets the socket name for tx affinity to driver.

        :returns: Gets the socket name for tx affinity to driver.
        :rtype: String
        """
        return self._txaffinity_to_driver_identity

    @property
    def rxaffinity_to_driver_identity(self):
        """Gets the socket name for rx affinity to driver.

        :returns: Gets the socket name for rx affinity to driver.
        :rtype: String
        """
        return self._rxaffinity_to_driver_identity

    @property
    def router_address(self):
        """
        Gets the socket address of the router that routes interprocess messages

        :return:    socket address of the router that routes interprocess messages
        :rtype:     String
        """
        return self._router_address


if __name__ == '__main__':
    SetAffinityOptions()
