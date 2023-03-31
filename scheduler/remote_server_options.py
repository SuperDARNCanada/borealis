#!/usr/bin/python3

"""
    remote_server_options.py
    ~~~~~~~~~~~~~~~~~~~~~~~
    options class for remote server module

    :copyright: 2019 SuperDARN Canada
"""

import os
import json

def ascii_encode_dict(data):
    return dict(map(lambda x: x.encode('ascii'), pair) for pair in data.items())


class RemoteServerOptions(object):
    """
    Parses the options from the config file that are relevant to data writing.

    """
    def __init__(self, config_path=None):
        """
        Initialize and get configuration options

        Args:
            config_path (str): path to config file for. Default BOREALISPATH/config/[rad]/[rad]_config.ini
        """
        super().__init__()

        # Gather the borealis configuration information
        # Gather the borealis configuration information
        if not os.environ["BOREALISPATH"]:
            raise ValueError("BOREALISPATH env variable not set")
        if not os.environ['RADAR_ID']:
            raise ValueError('RADAR_ID env variable not set')
        path = f'{os.environ["BOREALISPATH"]}/config/' \
            f'{os.environ["RADAR_ID"]}/' \
            f'{os.environ["RADAR_ID"]}_config.ini'
        try:
            with open(path, 'r') as data:
                raw_config = json.load(data)
        except IOError:
            print(f'IOError on config file at {path}')
            raise

        self._site_id = raw_config["site_id"]

    @property
    def site_id(self):
        """
        Gets the 3 letter radar code of this radar.

        :return:    3 letter radar code
        :rtype:     str
        """
        return self._site_id
