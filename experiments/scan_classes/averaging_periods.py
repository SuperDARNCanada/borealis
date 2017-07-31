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

    def __init__(self, ave_keys, ave_slice_dict, ave_interface, options, slice_to_beamorder_dict, slice_to_beamdir_dict):
        # make a list of the cpos in this AveragingPeriod.

        ScanClassBase.__init__(self, ave_keys, ave_slice_dict, ave_interface, options)

        self.slice_to_beamorder = slice_to_beamorder_dict
        self.slice_to_beamdir = slice_to_beamdir_dict

        # Metadata for an AveragingPeriod: clear frequency search, integration time, number of averages goal
        self.clrfrqf = False
        # there may be multiple slices in this averaging period at different frequencies so
        # we may have to search multiple ranges.
        self.clrfrqrange = []
        for slice_id in self.slice_ids:
            if self.slice_dict[slice_id]['clrfrqflag']:
                self.clrfrqf = True
                self.clrfrqrange.append(self.slice_dict[slice_id]['clrfrqrange'])

        # TODO: SET UP CLEAR FREQUENCY SEARCH CAPABILITY
        # also note for when setting this up clrfrqranges may overlap

        self.intt = self.slice_dict[self.slice_ids[0]]['intt']
        self.intn = self.slice_dict[self.slice_ids[0]]['intn']
        if self.intt is not None:  # intt has priority over intn
            for slice_id in self.slice_ids:
                if self.slice_dict[slice_id]['intt'] != self.intt:
                    errmsg = """Slice {} and {} are INTTIME mixed and do not have the
                        same Averaging Period duration intt""".format(self.slice_ids[0], slice_id)
                    raise ExperimentException(errmsg)
        elif self.intn is not None:
            for slice_id in self.slice_ids:
                if self.slice_dict[slice_id]['intn'] != self.intn:
                    errmsg = """Slice {} and {} are INTTIME mixed and do not have the
                        same NAVE goal intn""".format(self.slice_ids[0], slice_id)
                    raise ExperimentException(errmsg)

        for slice_id in self.slice_ids:
                if len(self.slice_dict[slice_id]['beam_order']) != len(self.slice_dict[self.slice_ids[0]]['beam_order']):
                    errmsg = """Slice {} and {} are INTTIME mixed and do not have the
                        same length of beam_order (number of integration periods)
                        """.format(self.slice_ids[0], slice_id)
                    raise ExperimentException(errmsg)

        # NOTE: Do not need beam information inside the AveragingPeriod, this is in Scan.

        # Determine how this averaging period is made by separating out
        #   the INTEGRATION mixed.

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
                integ_combos.append(list(k))

        combos = slice_combos_sorter(integ_combos, self.slice_ids)

        print("sequences slice id combos: {}".format(combos))

        return combos

    def set_beamdirdict(self, beamiter):
        """
        Get a dictionary of 'slice_id' : 'beamdir(s)' for this averaging period at a given beam iteration.
        :param beamiter: 
        :return: dictionary of slice to beamdir where beamdir is always a list (may be of length one though).
        Beamdir is azimuth angle.
        """
        slice_to_beamdir_dict = {}
        try:
            for slice_id in self.slice_ids:
                beam_number = self.slice_to_beamorder[slice_id][beamiter]
                if isinstance(beam_number, int):
                    beamdir = [self.slice_to_beamdir[slice_id][beam_number]]
                else:  # is a list
                    beamdir = [self.slice_to_beamdir[slice_id][bmnum] for bmnum in beam_number]
                slice_to_beamdir_dict[slice_id] = beamdir
        except IndexError:
            errmsg = 'Looking for BeamNumber or Beamdir that does not Exist at BeamIter' \
                     ' {}'.format(beamiter)
            raise ExperimentException(errmsg)

        return slice_to_beamdir_dict

    def build_sequences(self, slice_to_beamdir_dict, txctrfreq, txrate, options):  # TODO fix and input only options used or get from init
        """
        Build a list of sequences (lists of pulse dictionaries) containing all pulse samples data to iterate 
        through when
        transmitting. 
        :return: sequence_dict_list
        """
        # Create a pulse dictionary before running through the
        #   averaging period.
        sequence_dict_list = []
        # a list of sequence data.
        # a sequence data is a list of pulses in the sequence in order.
        # a pulse data is a dictionary of the required data for the pulse.
        for sequence in self.sequences:
            # create pulse dictionary.
            sequence_dict = sequence.build_pulse_transmit_data(slice_to_beamdir_dict, txctrfreq, txrate, options)

            # Just alternating sequences

            sequence_dict_list.append(sequence_dict)

            return sequence_dict_list
