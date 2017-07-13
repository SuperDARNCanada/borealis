#!/usr/bin/env python
# Copyright 2017 SuperDARN Canada
#
# Marci Detwiller
#
# Basic tests for use in checking slices.

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
