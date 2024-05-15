#!/usr/bin/env python

"""
    experiment_exception
    ~~~~~~~~~~~~~~~~~~~~
    This is the exception that is raised when there are problems with the experiment that
    cannot be remedied using experiment_prototype methods.

    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""


class ExperimentException(Exception):
    """
    Is raised for the exception where an experiment cannot be run due to setup errors.
    """

    def __init__(self, message, *args):
        self.message = message  # to avoid DeprecationWarning
        super(ExperimentException, self).__init__(message, *args)
