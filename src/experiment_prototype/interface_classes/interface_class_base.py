#!/usr/bin/python

"""
    interface_class_base
    ~~~~~~~~~~~~~~~~~~~~
    This is the base module for all InterfaceClassBase types (iterable for an experiment given certain
    parameters). These types include the Scan class, the AveragingPeriod class, and the Sequence
    class.

    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""
# built-in
import inspect
import itertools
from pathlib import Path

# third-party
import structlog

# local
from experiment_prototype.experiment_exception import ExperimentException

# Obtain the module name that imported this log_config
caller = Path(inspect.stack()[-1].filename)
module_name = caller.name.split(".")[0]
log = structlog.getLogger(module_name)


class InterfaceClassBase(object):
    """
    The base class for the classes Scan, AveragingPeriod, and Sequence. Scans are made up of
    AveragingPeriods, these are typically a 3sec time of the same pulse sequence pointing in one
    direction.  AveragingPeriods are made up of Sequences, typically the same sequence run ave.
    20-30 times after a clear frequency search.  Sequences are made up of pulses, which is a list of
    dictionaries where each dictionary describes a pulse.

    :param  object_keys:        slice_ids that need to be included in this interface_class_base type.
    :type   object_keys:        list
    :param  object_slice_dict:  the slice dictionary that explains the parameters of each slice that
                                is included in this interface_class_base type. Keys are the slice_ids
                                included and values are dictionaries including all necessary slice
                                parameters as keys.
    :type   object_slice_dict:  dict
    :param  object_interface:   the interfacing dictionary that describes how to interface the
                                slices that are included in this interface_class_base type. Keys are
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
                                - 'txrate' [Hz],
                                - 'intf_offset' [m,m,m],
                                - 'dm_rate'

    :type transmit_metadata:    dict
    """

    def __init__(
        self, object_keys, object_slice_dict, object_interface, transmit_metadata
    ):

        # list of slice_ids included in this interface_class_base
        self.slice_ids = object_keys

        # dictionary (key = slice_id) of dictionaries (value = slice parameters)
        self.slice_dict = object_slice_dict

        # interfacing dictionary (key = (slice_id_1, slice_id_2), value = one of
        # interface_types)
        self.interface = object_interface

        # The nested slice list is filled in a child class before the prep_for_nested_interface_class
        # function is run. This list is of format [[], [], ...] where the length of the outer list
        # is equal to the number of the lower interface_class_base instance within the instance of the
        # higher interface_class_base ( ex. number of sequences within averagingperiods)
        self.nested_slice_list = []

        # all necessary experiment-wide transmit metadata
        self.transmit_metadata = transmit_metadata

        # List of lists, each inner list is all slice ids that share a scan
        self.nested_slice_list = self.get_nested_slice_ids()

    def prep_for_nested_interface_class(self):
        """
        Retrieve the params needed for the nested class (also with base InterfaceClassBase).

        This class reduces duplicate code by breaking down the InterfaceClassBase class into the separate
        portions for the nested instances. For Scan class, the nested class is AveragingPeriod, and
        we will need to break down the parameters given to the Scan instance because there may be
        multiple AveragingPeriods within. For AveragingPeriod, the nested class is Sequence.

        :returns:   params for the nested class's instantiation.
        :rtype:     list
        """

        # TODO documentation make a detailed example of this and diagram
        nested_class_param_lists = []
        log.debug(self.nested_slice_list)
        for slice_list in self.nested_slice_list:
            slices_for_nested_class = {}
            for slice_id in slice_list:
                try:
                    slices_for_nested_class[slice_id] = self.slice_dict[slice_id]
                except KeyError:
                    errmsg = (
                        f"Error with slice list - slice id {slice_id} cannot be found."
                    )
                    raise ExperimentException(errmsg)

            # now take a subset of the interface dictionary that applies to this nested object
            # of interface_class_base type.
            nested_class_interface = {}
            for i in itertools.combinations(slice_list, 2):
                # slice_list is sorted so we should have the following effect:
                # combinations([1, 3, 5], 2) --> [1,3], [1,5], [3,5]
                nested_class_interface[tuple(i)] = self.interface[tuple(i)]

            nested_class_param_lists.append(
                [
                    slice_list,
                    slices_for_nested_class,
                    nested_class_interface,
                    self.transmit_metadata,
                ]
            )

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
        disjoint_sets = []

        # Go through all interfaces and create sets of mutually-interfaced slices
        for slice_interfacing in list_of_combos:
            slice_already_interfaced_with = False
            for disjoint_set in disjoint_sets:
                # Add both slice ids to this set if either one is already a member
                if (
                    slice_interfacing[0] in disjoint_set
                    or slice_interfacing[1] in disjoint_set
                ):
                    disjoint_set.update({slice_interfacing[0], slice_interfacing[1]})
                    slice_already_interfaced_with = True
            if not slice_already_interfaced_with:
                # Create a new set with just these two
                disjoint_sets.append({slice_interfacing[0], slice_interfacing[1]})

        # Check all slice ids and add any missing ones to their own set.
        for key in all_keys:
            slice_in_set = False
            for disjoint_set in disjoint_sets:
                if key in disjoint_set:  # This slice interfaces with others
                    slice_in_set = True
                    break
            if (
                not slice_in_set
            ):  # This slice doesn't interface with the others, make a new set
                disjoint_sets.append({key})

        # Go through all sets and make sure they are disjoint (don't share any slice ids)
        for i in range(len(disjoint_sets)):
            for j in range(i + 1, len(disjoint_sets)):
                bad_slices = disjoint_sets[i].intersection(disjoint_sets[j])
                if len(bad_slices) != 0:
                    raise ExperimentException(
                        f"The following slices do not interface well with other slices: "
                        f"{bad_slices}"
                    )
            disjoint_sets[i] = sorted(list(disjoint_sets[i]))  # Convert to a list

        disjoint_sets.sort(key=lambda x: x[0])  # Sort by the first key in each list
        return disjoint_sets

    def get_nested_slice_ids(self):
        """
        Organize the slice_ids by interface.

        This method is inherited by child classes and organizes all slices in each child class which
        should be combined by the class. For example, all slices in a Scan should be combined if they
        share an AveragingPeriod or Sequence or are concurrent.

        Returns a list of lists where each inner list contains the slices that are combined inside
        this object. e.g. for InterfaceClassBase:
        len(nested_slice_list) = # of scans in this experiment,
        len(nested_slice_list[0]) = # of slices in the first scan

        :returns:   A list that has one element per scan. Each element is a list of slice_ids
                    signifying which slices are combined inside that scan. The list returned could
                    be of length 1, meaning only one scan is present in the experiment.
        :rtype:     list of lists
        """
        nested_combos = []

        combine_below_dict = {
            "InterfaceClassBase": [
                "AVEPERIOD",
                "SEQUENCE",
                "CONCURRENT",
            ],  # Combine everything except SCAN interfaced
            "Scan": [
                "SEQUENCE",
                "CONCURRENT",
            ],  # Combine everything SEQUENCE or CONCURRENT interfaced
            "AveragingPeriod": [
                "CONCURRENT"
            ],  # Combine everything CONCURRENT interfaced
            "Sequence": [],  # All slices in a Sequence are already CONCURRENT and should be combined already
        }

        combine_list = combine_below_dict[
            type(self).__name__
        ]  # Returns the class name of the calling instance

        for slice_ids_combo, interface_value in self.interface.items():
            if interface_value in combine_list:
                nested_combos.append(list(slice_ids_combo))

        combos = self.slice_combos_sorter(nested_combos, self.slice_ids)

        return combos
