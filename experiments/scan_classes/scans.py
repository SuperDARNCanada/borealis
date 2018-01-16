#!/usr/bin/python

import sys
from experiments.scan_classes.averaging_periods import AveragingPeriod
from experiments.scan_classes.scan_class_base import ScanClassBase
from experiments.experiment_exception import ExperimentException


class Scan(ScanClassBase):
    """ 
    Made up of AveragingPeriods at defined beam directions, and some other metadata for the scan itself.
    
    Scans are made up of AveragingPeriods, these are typically a 3sec time of
    the same pulse sequence pointing in one direction.  AveragingPeriods are made
    up of Sequences, typically the same sequence run ave. 20-30 times after a clear
    frequency search.  Sequences are made up of pulses, which is a list of dictionaries 
    where each dictionary describes a pulse. 
    """

    def __init__(self, scan_keys, scan_slice_dict, scan_interface, options):

        ScanClassBase.__init__(self, scan_keys, scan_slice_dict, scan_interface, options)

        # scan metadata - must be the same between all slices combined in scan.  Metadata includes:
        self.scanboundflag = self.slice_dict[self.slice_ids[0]]['scanboundflag']
        for slice_id in self.slice_ids:
            if self.slice_dict[slice_id]['scanboundflag'] != self.scanboundflag:
                errmsg = """Scan Boundary Flag not the Same Between Slices {} and {} combined in Scan"""\
                    .format(self.slice_ids[0], slice_id)
                raise ExperimentException(errmsg)
        if self.scanboundflag:
            self.scanbound = self.slice_dict[self.slice_ids[0]]['scanbound']
            for slice_id in self.slice_ids:
                if self.slice_dict[slice_id]['scanbound'] != self.scanbound:
                    errmsg = """Scan Boundary not the Same Between Slices {} and {}
                         combined in Scan""".format(self.slice_ids[0], slice_id)
                    raise ExperimentException(errmsg)

        # NOTE: for now we assume that when INTTIME combined, the AveragingPeriods of the various slices in the scan are
        #   just interleaved 1 then the other.

        # Create a dictionary of beam directions for slice_id #
        self.beamdir = {}
        self.scan_beams = {}
        for slice_id in self.slice_ids:
            self.beamdir[slice_id] = self.slice_dict[slice_id]['beam_angle']
            self.scan_beams[slice_id] = self.slice_dict[slice_id]['beam_order']

        self.aveperiods = []
        self.nested_slice_list = self.get_inttime_slice_ids()

        for params in self.prep_for_nested_scan_class():
            self.aveperiods.append(AveragingPeriod(*params))

        # AveragingPeriod will be in slice_id # order

    def get_inttime_slice_ids(self):
        """
        Take the interface keys inside this scan and return a list of lists where each inner list 
        contains the slices that are in an inttime that is inside this scan. ie. len(combos) = 
        # of inttimes in this list, len(combos[0]) = # of slices in the first inttime, etc.
        :return: 
        """
        intt_combos = []

        for k, interface_value in self.interface.items():
            if (interface_value == "PULSE" or interface_value == "INTEGRATION"):
                intt_combos.append(list(k))
        # Inside the scan, we have a subset of the interface dictionary including all combinations
        # of slice_id that are included in this Scan instance. They could be interfaced INTTIME,
        # INTEGRATION, or PULSE. We want to remove all of the INTTIME combinations as we want to
        # eventually have a list of lists (combos) that is of length = # of INTTIMEs in the scan,
        # with all slices included in the inttimes inside the inner lists.

        # TODO make example and diagram

        combos = self.slice_combos_sorter(intt_combos, self.slice_ids)

        if __debug__:
            print("Inttime slice id list: {}".format(combos))

        return combos

    def prep_for_nested_scan_class(self):
        """
        Override to give more information about beamorder and beamdir
        :return: a list of lists of parameters that can be directly passed into the nested 
        ScanClassBase type, AveragingPeriod. the params_list is of length = # of AveragingPeriods
        in this scan. 
        """

        # Get the basic parameters for a ScanClassBase type
        params_list = ScanClassBase.prep_for_nested_scan_class(self)

        # Add the beam order and beam direction information that is necessary for AveragingPeriods
        # specifically.
        for params, inttime_list in zip(params_list, self.nested_slice_list):
            # Make sure the number of inttimes (as determined by length of slice['scan'] is the same
            # for slices combined in the averaging period.
            self.nested_beamorder = {}
            self.nested_beamdir = {}
            for slice_id in inttime_list:
                if len(self.scan_beams[slice_id]) != len(self.scan_beams[inttime_list[0]]):
                    errmsg = """Slice {} and {} are combined within the AveragingPeriod but do not 
                        have the same number of AveragingPeriods in their 
                        scan""".format(self.slice_ids[0], slice_id)
                    raise ExperimentException(errmsg)
                else:
                    self.nested_beamorder[slice_id] = self.scan_beams[slice_id]
                    self.nested_beamdir[slice_id] = self.beamdir[slice_id]
            params.append(self.nested_beamorder)
            params.append(self.nested_beamdir)

        if __debug__:
            print(params_list)
        return params_list
