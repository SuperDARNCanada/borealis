#!/usr/bin/python

# IB collab mode written by Devin Huyghebaert 20200609
import os
import sys

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from experiments.IB_collab_mode import IBCollabMode


class IBCollabMode_13(IBCollabMode):

    def __init__(self):

        super(IBCollabMode_13, self).__init__(freq=13000)
