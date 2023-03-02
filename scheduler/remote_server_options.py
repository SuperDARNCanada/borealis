#!/usr/bin/python3

"""
    remote_server_options.py
    ~~~~~~~~~~~~~~~~~~~~~~~
    options class for remote server module

    :copyright: 2019 SuperDARN Canada
"""

from src.utils.general import load_config


def ascii_encode_dict(data):
    ascii_encode = lambda x: x.encode('ascii')
    return dict(map(ascii_encode, pair) for pair in data.items())


class RemoteServerOptions(object):
    """
    Parses the options from the config file that are relevant to data writing.

    """
    def __init__(self):
        super(RemoteServerOptions, self).__init__()

        # Gather the borealis configuration information
        raw_config = load_config()

        self._site_id = raw_config["site_id"]

    @property
    def site_id(self):
        """
        Gets the 3 letter radar code of this radar.

        :return:    3 letter radar code
        :rtype:     str
        """
        return self._site_id


