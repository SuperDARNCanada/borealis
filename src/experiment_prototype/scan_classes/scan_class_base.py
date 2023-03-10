#!/usr/bin/python

"""
    scan_class_base
    ~~~~~~~~~~~~~~~
    This is the base module for all ScanClassBase types (iterable for an experiment given certain
    parameters). These types include the Scan class, the AveragingPeriod class, and the Sequence
    class.

    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""

from experiment_prototype.experiment_exception import ExperimentException
import itertools


class ScanClassBase(object):
    """
    The base class for the classes Scan, AveragingPeriod, and Sequence. Scans are made up of
    AveragingPeriods, these are typically a 3sec time of the same pulse sequence pointing in one
    direction.  AveragingPeriods are made up of Sequences, typically the same sequence run ave.
    20-30 times after a clear frequency search.  Sequences are made up of pulses, which is a list of
    dictionaries where each dictionary describes a pulse.

    :param  object_keys:        slice_ids that need to be included in this scan_class_base type.
    :type   object_keys:        list  
    :param  object_slice_dict:  the slice dictionary that explains the parameters of each slice that
                                is included in this scan_class_base type. Keys are the slice_ids
                                included and values are dictionaries including all necessary slice
                                parameters as keys.
    :type   object_slice_dict:  dict
    :param  object_interface:   the interfacing dictionary that describes how to interface the
                                slices that are included in this scan_class_base type. Keys are
                                tuples of format (slice_id_1, slice_id_2) and values are of
                                interface_types set up in experiment_prototype.
    :type   object_interface:   dict
    :param  transmit_metadata:  a dictionary of the experiment-wide transmit metadata for building
                                pulse sequences. The keys of the transmit_metadata are:

                                - 'output_rx_rate' [Hz],
                                - 'main_antenna_count',
                                - 'intf_antenna_count',
                                - 'tr_window_time' [s],
                                - 'main_antenna_spacing' [m],
                                - 'intf_antenna_spacing' [m],
                                - 'pulse_ramp_time' [s],
                                - 'max_usrp_dac_amplitude' [V peak],
                                - 'rx_sample_rate' [Hz],
                                - 'min_pulse_separation' [us],
                                - 'txctrfreq' [kHz],
                                - 'txrate' [Hz],
                                - 'intf_offset' [m,m,m],
                                - 'dm_rate'

    :type transmit_metadata:    dict
    """

    def __init__(self, object_keys, object_slice_dict, object_interface, transmit_metadata):

        # list of slice_ids included in this scan_class_base
        self.slice_ids = object_keys

        # dictionary (key = slice_id) of dictionaries (value = slice parameters)
        self.slice_dict = object_slice_dict

        # interfacing dictionary (key = (slice_id_1, slice_id_2), value = one of
        # interface_types)
        self.interface = object_interface

        # The nested slice list is filled in a child class before the prep_for_ nested_scan_class
        # function is run. This list is of format [[], [], ...] where the length of the outer list
        # is equal to the number of the lower scan_class_base instance within the instance of the
        # higher scan_class_base ( ex. number of sequences within averagingperiods)
        self.nested_slice_list = []

        # all necessary experiment-wide transmit metadata
        self.transmit_metadata = transmit_metadata

    def prep_for_nested_scan_class(self):
        """
        Retrieve the params needed for the nested class (also with base ScanClassBase).

        This class reduces duplicate code by breaking down the ScanClassBase class into the separate
        portions for the nested instances. For Scan class, the nested class is AveragingPeriod, and
        we will need to break down the parameters given to the Scan instance because there may be
        multiple AveragingPeriods within. For AveragingPeriod, the nested class is Sequence.

        :returns:   params for the nested class's instantiation.
        :rtype:     list
        """

        # TODO documentation make a detailed example of this and diagram
        nested_class_param_lists = []
        if __debug__:
            print(self.nested_slice_list)
        for slice_list in self.nested_slice_list:
            slices_for_nested_class = {}
            for slice_id in slice_list:
                try:
                    slices_for_nested_class[slice_id] = self.slice_dict[slice_id]
                except KeyError:
                    errmsg = f'Error with slice list - slice id {slice_id} cannot be found.'
                    raise ExperimentException(errmsg)

            # now take a subset of the interface dictionary that applies to this nested object
            # of scan_class_base type.
            nested_class_interface = {}
            for i in itertools.combinations(slice_list, 2):
                # slice_list is sorted so we should have the following effect:
                # combinations([1, 3, 5], 2) --> [1,3], [1,5], [3,5]
                nested_class_interface[tuple(i)] = self.interface[tuple(i)]

            nested_class_param_lists.append([slice_list, slices_for_nested_class,
                                             nested_class_interface, self.transmit_metadata])

        return nested_class_param_lists

    @staticmethod
    def slice_combos_sorter(list_of_combos, all_keys):
        """
        Sort keys of a list of combinations so that keys only appear once in the list.

        This function modifies the input list_of_combos so that all associated slices are in the
        same list. For example, if input is list_of_combos = [[0,1], [0,2], [0,4], [1,4], [2,4]] and
        all_keys = [0,1,2,4,5] then the output should be [[0,1,2,4], [5]]. This is used to get the
        slice dictionary for nested class instances. In the above example, we would then have two
        instances of the nested class to create: one with slices 0,1,2,4 and another with slice 5.

        :param      list_of_combos: list of lists of length two associating two slices together.
        :type       list_of_combos: list
        :param      all_keys:       list of all keys included in this object (scan, ave_period, or
                                    sequence).
        :type       all_keys:       list

        :returns:   list of combos that is sorted so that each key only appears once and the lists
                    within the list are of however long necessary
        :rtype:     list
        """

        list_of_combos = sorted(list_of_combos)

        # if [2,4] and [1,4], then also must be [1,2] in the list_of_combos Now we are going to
        # modify the list of lists of length = 2 to be a list of length x so that if [1,2] and [2,4]
        # and [1,4] are in list_of_combos, we want only one list element for this scan : [1,2,4] .

        scan_i = 0  # TODO detailed explanation with examples. Consider changing to a graph traversal algorithm?
        while scan_i < len(list_of_combos):  # i: element in list_of_combos (representing one scan)
            slice_id_k = 0
            while slice_id_k < len(list_of_combos[scan_i]):  # k: element in scan (representing a slice)
                scan_j = scan_i + 1  # j: iterates through the other elements of list_of_combos, to combine them into
                # the first, i, if they are in fact part of the same scan.
                while scan_j < len(list_of_combos):
                    if list_of_combos[scan_i][slice_id_k] == list_of_combos[scan_j][0]:
                        # if an element (slice_id) inside the i scan is the same as a slice_id in
                        # the j scan (somewhere further in the list_of_combos), then we need to
                        # combine that j scan into the i scan. We only need to check the first
                        # element of the j scan because list_of_combos has been sorted and we know
                        # the first slice_id in the scan is less than the second slice id.
                        add_n_slice_id = list_of_combos[scan_j][1]  # the slice_id to add to the i scan from the j scan.
                        list_of_combos[scan_i].append(add_n_slice_id)
                        # Combine the indices if there are 3+ slices combining in same scan
                        for m in range(0, len(list_of_combos[scan_i]) - 1):
                            # if we have added z to scan_i, such that scan_i is now [x,y,z], we now
                            # have to remove from the list_of_combos list [x,z], and [y,z].
                            # If x,z existed as SCAN but y,z did not, we have an error.

                            # Try all values in list_of_combos[i] except the last value, which is = to add_n.
                            try:
                                list_of_combos.remove([list_of_combos[scan_i][m], add_n_slice_id])
                                # list_of_combos[j][1] is the known last value in list_of_combos[i]
                            except ValueError:
                                # This error would occur if e.g. you had set [x,y] and [x,z] to
                                # CONCURRENT but [y,z] to SCAN. This means that we couldn't remove
                                # the scan_combo y,z from the list because it was not added to
                                # list_of_combos because it wasn't a scan type, so the interfacing
                                # would not make sense (conflict).
                                errmsg = f"Interfacing not Valid: exp_slice {list_of_combos[scan_i][m]}"\
                                    f" and exp_slice {list_of_combos[scan_i][slice_id_k]} are combined"\
                                    f" in-scan and do not interface the same with exp_slice {add_n_slice_id}"
                                raise ExperimentException(errmsg)
                        scan_j = scan_j - 1
                        # This means that the former list_of_combos[j] has been deleted and there
                        # are new values at index j, so decrement before incrementing in the while
                        # loop.
                        # The above for loop will delete more than one element of list_of_combos
                        # (min 2) but the while scan_j < len(list_of_combos) will reevaluate the
                        # length of list_of_combos.
                    scan_j = scan_j + 1
                slice_id_k = slice_id_k + 1  
                # if interfacing has been properly set up, the loop will only ever find elements to
                # add to scan_i when slice_id_k = 0. If there were errors though (ex. x,y and y,z =
                # CONCURRENT but x,z did not) then iterating through the slice_id elements will
                # allow us to find the error.
            scan_i = scan_i + 1  
            # At this point, all elements in the just-finished scan_i will not be found anywhere
            #  else in list_of_combos.

        # Now list_of_combos is a list of lists,  where a slice_id occurs only once, within the
        # nested list.

        for slice_id in all_keys:
            for combo in list_of_combos:
                if slice_id in combo:
                    break
            else:  # no break
                list_of_combos.append([slice_id])
                # Append the slice on its own, it is in its own object.

        list_of_combos = sorted(list_of_combos)
        return list_of_combos


