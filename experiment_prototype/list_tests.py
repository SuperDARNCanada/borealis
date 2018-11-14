#!/usr/bin/env python

"""
    list_tests
    ~~~~~~~~~~
    Basic tests for use in checking slices.
    
    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""


def is_increasing(list_to_check):
    """
    Check if list is increasing.
    
    :param list_to_check: a list of numbers
    :returns: boolean True if is increasing, False if not.
    """
    if not all(x < y for x, y in zip(list_to_check, list_to_check[1:])):
        return False
    else:
        return True


def has_duplicates(list_to_check):
    """
    Check if the list has duplicate values.
    
    :param list_to_check: A list to check.
    :returns: boolean True if duplicates exist, False if not.
    """
    no_duplicates = set()
    for element in list_to_check:
        if element not in no_duplicates:
            no_duplicates.add(element)
        else:
            return True
    else:  # no return yet
        return False


