#!/usr/bin/python3

"""
    remote_server_options.py
    ~~~~~~~~~~~~~~~~~~~~~~~
    options class for remote server module

    :copyright: 2019 SuperDARN Canada
"""

import os
import json
from dataclasses import dataclass, field


@dataclass
class RemoteServerOptions(object):
    """
    Parses the options from the config file that are relevant to data writing.

    """
    site_id: str = field(init=False)

    def __post_init__(self):
        if not os.environ["BOREALISPATH"]:
            raise ValueError("BOREALISPATH env variable not set")
        if not os.environ['RADAR_ID']:
            raise ValueError('RADAR_ID env variable not set')
        self.parse_config()  # Parse info from config file

    def parse_config(self):
        # Read in config.ini file for current site
        path = f'{os.environ["BOREALISPATH"]}/config/' \
               f'{os.environ["RADAR_ID"]}/' \
               f'{os.environ["RADAR_ID"]}_config.ini'
        try:
            with open(path, 'r') as data:
                raw_config = json.load(data)
        except IOError:
            print(f'IOError on config file at {path}')
            raise

        # Initialize all options from config file
        self.site_id = raw_config['site_id']
