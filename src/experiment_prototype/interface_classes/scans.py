#!/usr/bin/python

"""
    scans
    ~~~~~
    This is the module containing the Scan class. The Scan class contains the InterfaceClassBase members,
    as well as a scanbound (to be implemented), a beamdir dictionary and scan_beams dictionary which
    specify beam direction angle and beam order in a scan, respectively, for individual slices that
    are to be combined in this scan. Beam direction information gets passed on to the
    AveragingPeriod.

    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""
# built-in
import inspect
from pathlib import Path

# third-party
import structlog

# local
from experiment_prototype.interface_classes.averaging_periods import AveragingPeriod
from experiment_prototype.interface_classes.interface_class_base import (
    InterfaceClassBase,
)
from experiment_prototype.experiment_exception import ExperimentException

# Obtain the module name that imported this log_config
caller = Path(inspect.stack()[-1].filename)
module_name = caller.name.split(".")[0]
log = structlog.getLogger(module_name)


class Scan(InterfaceClassBase):
    """
    Set up the scans.

    A scan is made up of AveragingPeriods at defined beam directions, and some other metadata for
    the scan itself.

    **The unique members of the scan are (not a member of the interfaceclassbase):**

    scanbound
        A list of seconds past the minute for scans to align to. Must be increasing, and it is
        possible to have values greater than 60s.
    """

    def __init__(self, scan_keys, scan_slice_dict, scan_interface, transmit_metadata):

        InterfaceClassBase.__init__(
            self, scan_keys, scan_slice_dict, scan_interface, transmit_metadata
        )

        # scan metadata - must be the same between all slices combined in scan.  Metadata includes:
        self.scanbound = self.slice_dict[self.slice_ids[0]].scanbound
        for slice_id in self.slice_ids:
            if self.slice_dict[slice_id].scanbound != self.scanbound:
                errmsg = (
                    f"Scan boundary not the same between slices {self.slice_ids[0]} and"
                    f" {slice_id} for AVEPERIOD or CONCURRENT interfaced slices"
                )
                raise ExperimentException(errmsg)

        # NOTE: for now we assume that when AVEPERIOD combined, the AveragingPeriods of the various
        # slices in the scan are just interleaved 1 then the other.

        # Create a dictionary of beam directions for slice_id
        self.beamdir = {}
        self.scan_beams = {}
        for slice_id in self.slice_ids:
            self.beamdir[slice_id] = self.slice_dict[slice_id].beam_angle
            self.scan_beams[slice_id] = self.slice_dict[slice_id].rx_beam_order

        self.aveperiods = []
        self.nested_slice_list = self.get_nested_slice_ids()

        for params in self.prep_for_nested_interface_class():
            self.aveperiods.append(AveragingPeriod(*params))

        # determine how many beams in scan:
        num_unique_aveperiods = 0
        for aveperiod in self.aveperiods:
            num_unique_aveperiods += aveperiod.num_beams_in_scan

        self.num_unique_aveperiods = num_unique_aveperiods

        if self.scanbound:
            self.num_aveperiods_in_scan = len(self.scanbound)
            if self.num_unique_aveperiods == self.num_aveperiods_in_scan:
                # the number of beams to get through for all aveperiods in scan equals the length of
                # the scanbound, so we can align the iterations of the beams to the scanbound list
                # times.
                self.align_scan_to_beamorder = True
            else:
                # the number of beams to get through for all aveperiods is not equal to the number
                # of scanbound times, so the same beam will not always occur at the same scanbound
                # time.
                self.align_scan_to_beamorder = False
        else:
            self.num_aveperiods_in_scan = self.num_unique_aveperiods

        self.aveperiod_iter = 0  # used to keep track of index into aveperiods list.
        # AveragingPeriod will be in slice_id # order

    def prep_for_nested_interface_class(self):
        """
        Override of base method to give more information about beamorder and beamdir.

        Beam order and beamdir are required for instantiation of the nested class AveragingPeriod so
        we need to extract this information as well to fill self.aveperiods.

        :returns:   Parameters that can be directly passed into the nested InterfaceClassBase type,
                    AveragingPeriod. The params_list is of length = # of AveragingPeriods in this
                    scan.
        :rtype:     list of lists
        """

        # Get the basic parameters for a InterfaceClassBase type
        params_list = InterfaceClassBase.prep_for_nested_interface_class(self)

        # Add the beam order and beam direction information that is necessary for AveragingPeriods
        # specifically.
        for params, aveperiod_list in zip(params_list, self.nested_slice_list):
            # Make sure the number of averaging periods (as determined by length of slice['scan'])
            # is the same for slices combined in the averaging period.
            nested_beamorder = {}
            nested_beamdir = {}
            for slice_id in aveperiod_list:
                nested_beamorder[slice_id] = self.scan_beams[slice_id]
                nested_beamdir[slice_id] = self.beamdir[slice_id]
            params.append(nested_beamorder)
            params.append(nested_beamdir)

        return params_list
