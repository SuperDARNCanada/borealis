#!/usr/bin/python

"""
    averaging_periods
    ~~~~~~~~~~~~~~~~~
    This is the module containing the AveragingPeriod class. The AveragingPeriod class 
    contains the ScanClassBase members, as well as clrfrqflag (to be implemented), 
    intn (number of integrations to run), or intt(max time for integrations), 
    and it contains sequences of class Sequence.

    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""

""" Scans are made up of AveragingPeriods, these are typically a 3sec time of
the same pulse sequence pointing in one direction.  AveragingPeriods are made
up of Sequences, typically the same sequence run ave. 20-30 times after a clear
frequency search.  Sequences are made up of pulse_time lists, which give
timing, slice, and pulsenumber. CPObject provides channels, pulseshift (if
any), freq, pulse length, beamdir, and wavetype.
"""

import sys
import operator

from experiment_prototype.scan_classes.sequences import Sequence
from experiment_prototype.scan_classes.scan_class_base import ScanClassBase
from experiment_prototype.experiment_exception import ExperimentException


class AveragingPeriod(ScanClassBase):
    """ 
    Set up the AveragingPeriods.
    
    An averagingperiod contains sequences and integrates one or multiple pulse sequences 
    together in a given time frame or in a given number of averages, if that is the 
    preferred limiter.
    
    **The unique members of the averagingperiod are (not a member of the scanclassbase):**
    
    slice_to_beamorder
        passed in by the scan that this AveragingPeriod instance is contained in. A 
        dictionary of slice: beam_order for all slices contained in this aveperiod.
    slice_to_beamdir
        passed in by the scan that this AveragingPeriod instance is contained in. A 
        dictionary of slice: beamdir(s) for all slices contained in this aveperiod.
    clrfrqflag
        Boolean, True if clrfrqsearch should be performed.
    clrfrqrange
        The range of frequency to search if clrfrqflag is True.  Otherwise empty.
    intt
        The priority limitation. The time limit (ms) at which time the aveperiod will 
        end. If None, we will use intn to end the aveperiod (a number of sequences).
    intn
        Number of averages (# of times the sequence transmits) to end after for the 
        averagingperiod. 
    sequences
        The list of sequences included in this aveperiod. This does not indicate how
        many averages will be transmitted in the aveperiod. If there are multiple 
        sequences in the list, they will be alternated between until the end of the 
        aveperiod.
    one_pulse_only
        boolean, True if this averaging period only has one unique set of pulse samples in it.
        This is true if there is only one sequence in the averaging period, and all pulses after the
        first pulse in the sequence have the isarepeat key = True. This boolean can be used to
        speed up the process of sending data to the driver which means we can get more averages
        in less time.
    """

    def __init__(self, ave_keys, ave_slice_dict, ave_interface, transmit_metadata,
                 slice_to_beamorder_dict, slice_to_beamdir_dict):

        ScanClassBase.__init__(self, ave_keys, ave_slice_dict, ave_interface, transmit_metadata)

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
                    errmsg = "Slices {} and {} are INTEGRATION or PULSE interfaced and do not have the" \
                             " same Averaging Period duration intt".format(self.slice_ids[0], slice_id)
                    raise ExperimentException(errmsg)
        elif self.intn is not None:
            for slice_id in self.slice_ids:
                if self.slice_dict[slice_id]['intn'] != self.intn:
                    errmsg = "Slices {} and {} are INTEGRATION or PULSE interfaced and do not have the" \
                             " same NAVE goal intn".format(self.slice_ids[0], slice_id)
                    raise ExperimentException(errmsg)

        for slice_id in self.slice_ids:  # TODO: This test seems to be superseded by the test for scan boundary being the same between slices
            if len(self.slice_dict[slice_id]['beam_order']) != len(self.slice_dict[self.slice_ids[0]]['beam_order']):
                errmsg = "Slices {} and {} are INTEGRATION or PULSE interfaced and do not have the" \
                         " same length of beam_order (number of integration periods)" \
                         .format(self.slice_ids[0], slice_id)
                raise ExperimentException(errmsg)

        # NOTE: Do not need beam information inside the AveragingPeriod, this is in Scan.

        # Determine how this averaging period is made by separating out the INTEGRATION interfaced.
        self.nested_slice_list = self.get_sequence_slice_ids()
        self.sequences = []

        for params in self.prep_for_nested_scan_class():
            self.sequences.append(Sequence(*params))

        self.one_pulse_only = False


    def get_sequence_slice_ids(self):
        """
        Return the slice_ids that are within the Sequences in this AveragingPeriod 
        instance.
        
        Take the interface keys inside this averagingperiod and return a list of lists 
        where each inner list contains the slices that are in a sequence that is inside 
        this averagingperiod. ie. len(nested_slice_list) = # of sequences in this 
        averagingperiod, len(nested_slice_list[0]) = # of slices in the first sequence, 
        etc.
        
        :returns: the nested_slice_list which is used when creating the sequences in 
         this averagingperiod.
        """

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
        Get a dictionary of 'slice_id' : 'beamdir(s)' for this averaging period.
        
        At a given beam iteration, this averagingperiod instance will select the beam 
        directions that it will shift to. 
        
        :param beamiter: the index into the beam_order list, or the index of an averaging
         period into the scan 
        :returns: dictionary of slice to beamdir where beamdir is always a list (may be 
         of length one though). Beamdir is azimuth angle.
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

    def build_sequences(self, slice_to_beamdir_dict):
        """
        Build a list of sequences to iterate through when transmitting.
         
        This includes building all pulses within the sequences, so it then contains all 
        pulse samples data to iterate through when transmitting. If there is only one 
        sequence type in the averaging period, this list will be of length 1. That 
        would mean that that one sequence gets repeated throughout the averagingperiod 
        (intn and intt still apply).
        
        :returns: sequence_dict_list, list of lists of pulse dictionaries. 
        """

        # Create a pulse dictionary before running through the
        #   averaging period.
        sequence_dict_list = []
        # a list of sequence data.
        # a sequence data is a list of pulses in the sequence in order.
        # a pulse data is a dictionary of the required data for the pulse.
        for sequence in self.sequences:
            # create pulse dictionary.
            sequence_dict = sequence.build_pulse_transmit_data(slice_to_beamdir_dict)

            # Just alternating sequences

            sequence_dict_list.append(sequence_dict)

        if len(sequence_dict_list) == 1:
            # only one sequence in the averaging period
            for pulse_num, pulse_dict in enumerate(sequence_dict_list[0]):
                if pulse_num == 0:
                    continue
                elif pulse_dict['isarepeat'] == False:
                    break  # there is another unique pulse in the sequence.
            else:  # no break
                self.one_pulse_only = True  # there is only one unique pulse in the sequence.

        return sequence_dict_list
