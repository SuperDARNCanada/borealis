"""
    general utilities
    ~~~~~~~~~~~~~~~~~

    General purpose cross functional functions

    :copyright: 2023 SuperDARN Canada
    :author: Adam Lozinsky
"""

import json
import os


def load_config():
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
            config = json.load(data)
    except IOError:
        print(f'IOError on config file at {path}')
        raise

    return config
