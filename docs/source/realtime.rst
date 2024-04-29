.. _realtime-package:

================
Realtime Package
================

The realtime package contains the utilities that check data files for corruption during operation
and serving the data to external sockets.

This module is optional, and can be disabled when running the radar by an argument to ``steamed_hams.py``. For example::

    /home/radar/borealis/scripts/steamed_hams.py twofsound release --realtime-off

Additional data processing is done using a Python package called ``backscatter``, which is an implementation of the
standard FITACF3 algorithm for analyzing SuperDARN data. More details about this package can be found at its GitHub
`homepage <https://github.com/SuperDARNCanada/backscatter>`_.
