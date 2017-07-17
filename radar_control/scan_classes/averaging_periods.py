#!/usr/bin/python
# REVIEW 1 documentation needs to be updated. Also, can be very verbose and descriptive of these classes. TODO
# Add the detailed description of each class into the docstring of the class instead of top of file.
""" Scans are made up of AveragingPeriods, these are typically a 3sec time of
the same pulse sequence pointing in one direction.  AveragingPeriods are made
up of Sequences, typically the same sequence run ave. 21 times after a clear
frequency search.  Sequences are made up of pulse_time lists, which give
timing, CPObject, and pulsenumber. CPObject provides channels, pulseshift (if
any), freq, pulse length, beamdir, and wavetype.
"""

import sys
import operator

from sequences import Sequence
from experiments.list_tests import slice_combos_sorter
from scan_class_base import ScanClassBase
from experiments.experiment_exception import ExperimentException

class AveragingPeriod(ScanClassBase):
    """ Made up of multiple pulse sequences (integrations) for one
    integration time.

    #REVIEW #1 "Integrates multiple pulse sequences together in a given time frame." sounds a bit more
    clear.
    """

    def __init__(self, ave_keys, ave_slice_dict, ave_interface, options):
        # make a list of the cpos in this AveragingPeriod.

        ScanClassBase.__init__(self, ave_keys, ave_slice_dict, ave_interface, options)

        # Metadata for an AveragingPeriod: clear frequency search, integration time, number of averages goal
        self.clrfrqf = False
        # there may be multiple slices in this averaging period at different frequencies so
        # we may have to search multiple ranges.
        self.clrfrqrange = []
        for slice_id in self.keys:
            if self.slice_dict[slice_id]['clrfrqflag']:
                self.clrfrqf = True
                self.clrfrqrange.append(self.slice_dict[slice_id]['clrfrqrange'])

        # TODO: SET UP CLEAR FREQUENCY SEARCH CAPABILITY
        # also note for when setting this up clrfrqranges may overlap

        self.intt = self.slice_dict[self.keys[0]]['intt']
        self.intn = self.slice_dict[self.keys[0]]['intn']
        if self.intt is not None:  # intt has priority over intn
            for slice_id in self.keys:
                if self.slice_dict[slice_id]['intt'] != self.intt:
                    errmsg = """Slice {} and {} are INTTIME mixed and do not have the
                        same Averaging Period duration intt""".format(self.keys[0], slice_id)
                    raise ExperimentException(errmsg)
        elif self.intn is not None:
            for slice_id in self.keys:
                if self.slice_dict[slice_id]['intn'] != self.intn:
                    errmsg = """Slice {} and {} are INTTIME mixed and do not have the
                        same NAVE goal intn""".format(self.keys[0], slice_id)
                    raise ExperimentException(errmsg)

        # NOTE: Do not need beam information inside the AveragingPeriod, this is in Scan.

        # Determine how this averaging period is made by separating out
        #   the INTEGRATION mixed.

        # NOTE: There is duplicate code here between scans, averagingperiods and sequences.
        self.slice_sequence_list = self.get_sequences()
        self.sequences = []
        self.nested_slice_list = self.slice_sequence_list

        for params in self.prep_for_nested_scan_class():
            self.sequences.append(Sequence(*params))

    def get_sequences(self):
        integ_combos = []

        # Remove INTEGRATION combos as we are trying to separate those.
        for k in self.interface.keys():
            if self.interface[k] == "PULSE":
                integ_combos.append([k])

        combos = slice_combos_sorter(integ_combos, self.keys)

        return combos
