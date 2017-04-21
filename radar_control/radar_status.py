

import sys
import os


def statustype(): # REVIEW #39 since we don't really need the numbers here, can we just make this a tuple of strings?
    return { 0 : 'CPNEEDED', 1 : 'NOERROR', 2 : 'WARNING', 3 : 'EXITERROR' }


def errortype():
    return {} # REVIEW #6 TODO?


class RadarStatus(): # REVIEW #6 TODO to finish the class?
    """Class to define transmit specifications of a certain frequency, 
    beam, and pulse sequence.
    """

    def __init__(self):
        self.status = 0 # needs a control program.
        self.errorcode = None
        


