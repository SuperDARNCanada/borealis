#!/usr/bin/python

""" Scans are made up of AveragingPeriods, these are typically a 3sec time of
the same pulse sequence pointing in one direction.  AveragingPeriods are made
up of Sequences, typically the same sequence run ave. 21 times after a clear
frequency search.  Sequences are made up of pulse_time lists, which give
timing, CPObject, and pulsenumber. CPObject provides channels, pulseshift (if
any), freq, pulse length, beamdir, and wavetype.
"""

import sys
import averaging_periods
from experiments.list_tests import slice_combos_sorter
from experiments.experiment_exception import ExperimentException


class ScanClassBase(object):
    """
    The base class for the classes scan, ave_period, and sequence.
    """

    def __init__(self, object_keys, object_slice_dict, object_interface, options):
        self.slice_ids = object_keys  # slice ids included in this object.
        self.slice_dict = object_slice_dict
        self.interface = object_interface
        self.nested_slice_list = []
        self.options = options

    def prep_for_nested_scan_class(self):
        """
        This class reduces duplicate code by breaking down the ScanClassBase class into the 
        separate portions for the nested class. For Scan class, the nested class is AveragingPeriod, 
        and we will need to break down the parameters given to the Scan instance because there may
        be multiple AveragingPeriods within. For AveragingPeriod, the nested class is Sequence.
        :return: params for the nested class's instantiation.
        """
        nested_class_param_lists = []
        print self.nested_slice_list
        for slice_list in self.nested_slice_list:
            slices_for_nested_class = {}
            for slice_id in slice_list:
                try:
                    slices_for_nested_class[slice_id] = self.slice_dict[slice_id]
                except KeyError:
                    errmsg = 'Error with slice list - slice id {} cannot be found.'.format(slice_id)
                    raise ExperimentException(errmsg)
            nested_interface_keys = []
            for m in range(len(slice_list)):
                for n in range(m + 1, len(slice_list)):
                    nested_interface_keys.append(tuple([slice_list[m], slice_list[n]]))
            nested_class_interface = {}
            for k in nested_interface_keys:
                nested_class_interface[k] = self.interface[k]

            nested_class_param_lists.append([slice_list, slices_for_nested_class,
                                             nested_class_interface, self.options])

        return nested_class_param_lists
