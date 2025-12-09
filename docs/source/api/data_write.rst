.. _data_write:

==========
Data Write
==========

The data_write package contains the utilities to parse messages containing antennas_iq data,
bfiq data, rawacf data, etc., and write that data to HDF5 or DMAP files.

For ``antennas_iq`` and ``bfiq`` files, only HDF5 files are supported. For ``rawacf`` files, you can choose to
write to either to disk in either the HDF5 or DMAP file formats, configuring this in the radar's
:ref:`Config File<config-options>`.

Usage
-----

.. argparse::
   :module: src.data_write
   :func: dw_parser
   :prog: data_write.py

.. automodule:: src.data_write
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: dw_parser
