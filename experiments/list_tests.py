#!/usr/bin/env python
# Copyright 2017 SuperDARN Canada
#
# Marci Detwiller
#
# Basic tests for use in checking slices.

from experiments.experiment_exception import ExperimentException


def is_increasing(list_to_check):
    """
    Return True if list is increasing, False if it is not.
    :param list_to_check: a list of numbers
    :return: boolean
    """
    if not all(x < y for x, y in zip(list_to_check, list_to_check[1:])):
        return False
    else:
        return True


def has_duplicates(list_to_check):
    """
    Return True if there are duplicates in the list, False if not.
    :param list_to_check: A list to check.
    :return: boolean. 
    """
    no_duplicates = set()
    for element in list_to_check:
        if element not in no_duplicates:
            no_duplicates.add(element)
        else:
            return True
    else: # no return yet
        return False


