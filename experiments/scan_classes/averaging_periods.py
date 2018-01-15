#!/usr/bin/python
# TODO be very descriptive of these classes in explanations

""" Scans are made up of AveragingPeriods, these are typically a 3sec time of
the same pulse sequence pointing in one direction.  AveragingPeriods are made
up of Sequences, typically the same sequence run ave. 20-30 times after a clear
frequency search.  Sequences are made up of pulse_time lists, which give
timing, slice, and pulsenumber. CPObject provides channels, pulseshift (if
any), freq, pulse length, beamdir, and wavetype.
"""

import sys
import operator

from sequences import Sequence
from scan_class_base import ScanClassBase
from experiments.experiment_exception import ExperimentException


class AveragingPeriod(ScanClassBase):
    """ 
    Scans are made up of AveragingPeriods, these are typically a 3sec time of
    the same pulse sequence pointing in one direction.  AveragingPeriods are made
    up of Sequences, typically the same sequence run ave. 20-30 times after a clear
    frequency search.  Sequences are made up of pulses, which is a list of dictionaries 
    where each dictionary describes a pulse. 
    
    Integrates multiple pulse sequences together in a given time frame or in a given number
    of averages, if that is the preferred limiter.
    """

    def __init__(self, ave_keys, ave_slice_dict, ave_interface, options, slice_to_beamorder_dict,
                 slice_to_beamdir_dict):

        ScanClassBase.__init__(self, ave_keys, ave_slice_dict, ave_interface, options)

        self.slice_to_beamorder = slice_to_beamorder_dict
        self.slice_to_beamdir = slice_to_beamdir_dict

        # Metadata for an AveragingPeriod: clear frequency search, integration time, number of averages goal
        self.clrfrqflag = False
        # there may be multiple slices in this averaging period at different frequencies so
        # we may have to search multiple ranges.
        self.clrfrqrange = []
        for slice_id in self.slice_ids:
            if self.slice_dict[slice_id]['clrfrqflag']:
                self.clrfrqflag = True
                self.clrfrqrange.append(self.slice_dict[slice_id]['clrfrqrange'])

        # TODO: SET UP CLEAR FREQUENCY SEARCH CAPABILITY
        # also note for when setting this up clrfrqranges may overlap

        self.intt = self.slice_dict[self.slice_ids[0]]['intt']
        self.intn = self.slice_dict[self.slice_ids[0]]['intn']
        if self.intt is not None:  # intt has priority over intn
            for slice_id in self.slice_ids:
                if self.slice_dict[slice_id]['intt'] != self.intt:
                    errmsg = """Slice {} and {} are INTTIME interfaced and do not have the
                        same Averaging Period duration intt""".format(self.slice_ids[0], slice_id)
                    raise ExperimentException(errmsg)
        elif self.intn is not None:
            for slice_id in self.slice_ids:
                if self.slice_dict[slice_id]['intn'] != self.intn:
                    errmsg = """Slice {} and {} are INTTIME interfaced and do not have the
                        same NAVE goal intn""".format(self.slice_ids[0], slice_id)
                    raise ExperimentException(errmsg)

        for slice_id in self.slice_ids:
                if len(self.slice_dict[slice_id]['beam_order']) != len(self.slice_dict[self.slice_ids[0]]['beam_order']):
                    errmsg = """Slice {} and {} are INTTIME interfaced and do not have the
                        same length of beam_order (number of integration periods)
                        """.format(self.slice_ids[0], slice_id)
                    raise ExperimentException(errmsg)

        # NOTE: Do not need beam information inside the AveragingPeriod, this is in Scan.

        # Determine how this averaging period is made by separating out the INTEGRATION interfaced.
        # TODO removed slice_sequence_list here ; didn't break anything? (was = nested_slice_list)
        self.nested_slice_list = self.get_sequence_slice_ids()
        self.sequences = []

        for params in self.prep_for_nested_scan_class():
            self.sequences.append(Sequence(*params))

    def get_sequence_slice_ids(self):
        integ_combos = []

        # Remove INTEGRATION combos as we are trying to separate those.
        for k, interface_type in self.interface.items():  # TODO make example
            if interface_type == "PULSE":
                integ_combos.append(list(k))

        combos = self.slice_combos_sorter(integ_combos, self.slice_ids)

        if __debug__:
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
                    beamdir = []
                    beamdir.append(self.slice_to_beamdir[slice_id][beam_number])
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
