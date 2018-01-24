# Copyright 2017 SuperDARN Canada
#
# Marci Detwiller
#
# Exceptions to be raised when the experiment is not properly set up or
# encounters any error causing it to break or modify itself.


class ExperimentException(Exception):
    """
    Raise for the exception where an experiment cannot be run due to setup errors. This is the 
    base class.
    """
    def __init__(self, message, *args):
        self.message = message  # to avoid DeprecationWarning
        super(ExperimentException, self).__init__(message, *args)
