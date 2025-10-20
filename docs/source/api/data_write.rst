==========
Data Write
==========

The data_write package contains the utilities to parse messages containing antennas_iq data,
bfiq data, rawacf data, etc., and write that data to HDF5 or DMAP files.

For ``antennas_iq`` and ``bfiq`` files, only HDF5 files are supported. For ``rawacf`` files, you can choose to
write to either to disk in either the HDF5 or DMAP file formats, configuring this in the radar's
:ref:`Config File<config-options>`.

Starting the ``data_write`` module::

    data_write.py [-h] [--enable-raw-acfs] [--enable-bfiq] [--enable-antenna-iq] [--enable-raw-rf] [--rawacf-format {hdf5,dmap}]

    Write processed SuperDARN data to file

    options:
      -h, --help            show this help message and exit
      --enable-raw-acfs     Enable raw acf writing
      --enable-bfiq         Enable beamformed iq writing
      --enable-antenna-iq   Enable individual antenna iq writing
      --enable-raw-rf       Save raw, unfiltered IQ samples. Requires HDF5.
      --rawacf-format {hdf5,dmap}
                            Format to store rawacf files in.

.. automodule:: src.data_write
    :members:
    :undoc-members:
    :show-inheritance:
