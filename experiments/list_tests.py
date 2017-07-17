#!/usr/bin/env python
# Copyright 2017 SuperDARN Canada
#
# Marci Detwiller
#
# Basic tests for use in checking slices.

from experiments.experiment_exception import ExperimentException


def isincreasing(list_to_check):

    if not all(x < y for x, y in zip(list_to_check, list_to_check[1:])):
        return False
    else:
        return True


def isduplicates(list_to_check):

    no_duplicates = set()
    for element in list_to_check:
        if element not in no_duplicates:
            no_duplicates.add(element)
        else:
            return True
    else: # no return yet
        return False


def slice_combos_sorter(list_of_combos, all_keys):
    """
    This function modifes the input list_of_combos so that all slices that are associated are 
    associated in the same list. For example, if input is list_of_combos = 
    [[0,1], [0,2], [0,4], [1,4], [2,4]] and all_keys = [0,1,2,4,5] then the output should be 
    [[0,1,2,4], [5]]. 
    :param list_of_combos: list of lists of length two associating two slices together. 
    :param all_keys: list of all keys included in this object (scan, ave_period, or sequence).
    :return: list of combos that is sorted so that each key only appears once and the lists within
     the list are of however long necessary
    """

    list_of_combos = sorted(list_of_combos)

    # if [2,4] and [1,4], then also must be [1,2] in the list_of_combos
    # Now we are going to modify the list of lists of length = 2 to be a list of length x so that if [1,2] and [2,4]
    # and [1,4] are in list_of_combos, we want only one list element for this scan : [1,2,4] .

    scan_i = 0  # TODO detailed explanation with examples.
    while scan_i < len(list_of_combos):  # i: element in list_of_combos (representing one scan)
        slice_id_k = 0
        while slice_id_k < len(list_of_combos[scan_i]):  # k: element in scan (representing a slice)
            scan_j = scan_i + 1  # j: iterates through the other elements of list_of_combos, to combine them into
            # the first, i, if they are in fact part of the same scan.
            while scan_j < len(list_of_combos):
                if list_of_combos[scan_i][slice_id_k] == list_of_combos[scan_j][0]:  # if an element (slice_id) inside
                    # the i scan is the same as a slice_id in the j scan (somewhere further in the list_of_combos),
                    # then we need to combine that j scan into the i scan. We only need to check the first element
                    # of the j scan because list_of_combos has been sorted and we know the first slice_id in the scan
                    # is less than the second slice id.
                    add_n_slice_id = list_of_combos[scan_j][1]  # the slice_id to add to the i scan from the j scan.
                    list_of_combos[scan_i].append(add_n_slice_id)
                    # Combine the indices if there are 3+ slices combining in same scan
                    for m in range(0, len(list_of_combos[scan_i]) - 1):  # if we have added z to scan_i, such that
                        # scan_i is now [x,y,z], we now have to remove from the list_of_combos list [x,z], and [y,z].
                        # If x,z existed as SCAN but y,z did not, we have an error.
                        # Try all values in list_of_combos[i] except the last value, which is = to add_n.
                        try:
                            list_of_combos.remove([list_of_combos[scan_i][m], add_n_slice_id])
                            # list_of_combos[j][1] is the known last value in list_of_combos[i]
                        except ValueError:
                            # This error would occur if you had set [x,y] and [x,z] to PULSE but [y,z] to
                            # SCAN. This means that we couldn't remove the scan_combo y,z from the list because it
                            # was not added to list_of_combos because it wasn't a scan type, so the interfacing would
                            # not make sense (conflict).
                            errmsg = 'Interfacing not Valid: exp_slice {} and exp_slice {} are combined in-scan and do not \
                                interface the same with exp_slice {}'.format(
                                list_of_combos[scan_i][m],
                                list_of_combos[scan_i][slice_id_k],
                                add_n_slice_id)
                            raise ExperimentException(errmsg)
                    scan_j = scan_j - 1
                    # This means that the former list_of_combos[j] has been deleted and there are new values at
                    #   index j, so decrement before incrementing in while.
                    # The above for loop will delete more than one element of list_of_combos (min 2) but the
                    # while scan_j < len(list_of_combos) will reevaluate the length of list_of_combos.
                scan_j = scan_j + 1
            slice_id_k = slice_id_k + 1  # if interfacing has been properly set up, the loop will only ever find
            # elements to add to scan_i when slice_id_k = 0. If there were errors though (ex. x,y and y,z = PULSE
            # but x,z did not) then iterating through the slice_id elements will allow us to find the
            # error.
        scan_i = scan_i + 1  # At this point, all elements in the just-finished scan_i will not be found anywhere
        #  else in list_of_combos.

    # Now list_of_combos is a list of lists,  where a slice_id occurs only once, within the nested list.

    for slice_id in all_keys:
        for combo in list_of_combos:
            if slice_id in combo:
                break
        else:  # no break
            list_of_combos.append([slice_id])
            # Append the slice on its own, it is in its own object.

    list_of_combos = sorted(list_of_combos)
    return list_of_combos
