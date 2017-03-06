

import sys
import os


def statustype():
    return { 0 : 'CPNEEDED', 1 : 'NOERROR', 2 : 'WARNING', 3 : 'EXITERROR' }


def errortype():
    return {}


class RadarStatus:
    """Class to define transmit specifications of a certain frequency, 
    beam, and pulse sequence.
    """

    def __init__(self):
        self.status = 0 # needs a control program.
        self.errorcode = None
        


