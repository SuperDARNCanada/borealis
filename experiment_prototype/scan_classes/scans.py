#!/usr/bin/python

"""
    scans
    ~~~~~
    This is the module containing the Scan class. The Scan class contains the
    ScanClassBase members, as well as a scanbound (to be implemented), a beamdir
    dictionary and scan_beams dictionary which specify beam direction angle and beam
    order in a scan, respectively, for individual slices that are to be combined in this
    scan. Beam direction information gets passed on to the AveragingPeriod.

    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""

import sys
from experiment_prototype.scan_classes.averaging_periods import AveragingPeriod
from experiment_prototype.scan_classes.scan_class_base import ScanClassBase
from experiment_prototype.experiment_exception import ExperimentException


class Scan(ScanClassBase):
    """
    Set up the scans.

    A scan is made up of AveragingPeriods at defined beam directions, and some other
    metadata for the scan itself.

    **The unique members of the scan are (not a member of the scanclassbase):**


    scanbound
        A list of seconds past the minute for scans to align to. Must be increasing,
        and it is possible to have values greater than 60s.
    """

    def __init__(self, scan_keys, scan_slice_dict, scan_interface, transmit_metadata):

        ScanClassBase.__init__(self, scan_keys, scan_slice_dict, scan_interface, transmit_metadata)

        # scan metadata - must be the same between all slices combined in scan.  Metadata includes:
        self.scanbound = self.slice_dict[self.slice_ids[0]]['scanbound']
        for slice_id in self.slice_ids:
            if self.slice_dict[slice_id]['scanbound'] != self.scanbound:
                errmsg = "Scan boundary not the same between slices {} and {}" \
                         " for INTTIME or PULSE interfaced slices".format(self.slice_ids[0], slice_id)
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
        Return the slice_ids that are within the AveragingPeriods in this Scan instance.

        Take the interface keys inside this scan and return a list of lists
        where each inner list contains the slices that are in an averagingperiod that is
        inside this scan. ie. len(nested_slice_list) = # of averagingperiods in this
        scan, len(nested_slice_list[0]) = # of slices in the first averagingperiod,
        etc.

        :returns: the nested_slice_list which is used when creating the
         AveragingPeriods for this scan.
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
        Override of base method to give more information about beamorder and beamdir.

        Beam order and beamdir are required for instantiation of the nested class
        AveragingPeriod so we need to extract this information as well to fill
        self.aveperiods.

        :returns: a list of lists of parameters that can be directly passed into the
         nested ScanClassBase type, AveragingPeriod. the params_list is of length = # of
         AveragingPeriods in this scan.
        """

        # Get the basic parameters for a ScanClassBase type
        params_list = ScanClassBase.prep_for_nested_scan_class(self)

        # Add the beam order and beam direction information that is necessary for
        # AveragingPeriods specifically.
        for params, inttime_list in zip(params_list, self.nested_slice_list):
            # Make sure the number of inttimes (as determined by length of slice['scan']
            # is the same for slices combined in the averaging period.
            self.nested_beamorder = {}
            self.nested_beamdir = {}
            for slice_id in inttime_list:
                if len(self.scan_beams[slice_id]) != len(self.scan_beams[inttime_list[0]]):
                    errmsg = """Slice {} and {} are INTEGRATION or PULSE interfaced but do not
                        have the same number of integrations in their
                        scan""".format(self.slice_ids[0], slice_id)
                    raise ExperimentException(errmsg)
                else:
                    self.nested_beamorder[slice_id] = self.scan_beams[slice_id]
                    self.nested_beamdir[slice_id] = self.beamdir[slice_id]
            params.append(self.nested_beamorder)
            params.append(self.nested_beamdir)

        return params_list
