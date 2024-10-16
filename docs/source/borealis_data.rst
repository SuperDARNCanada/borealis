===================
Borealis Data Files
===================

---------------
Data Generation
---------------

The Borealis software module ``data_write.py`` is responsible for writing all data files. Different
flags can be given to the module to write various types of files. See the documentation for
:ref:`Borealis Processes`

Borealis writes files into `HDF5 format <https://portal.hdfgroup.org/display/support>`_. Examples of
how to use HDF5 files can be found here for multiple languages: `HDF5 Examples
<https://portal.hdfgroup.org/display/HDF5/HDF5+Examples>`_

The following data file types can be generated by Borealis in HDF5 format. The standard Borealis
``release`` mode run by the scheduler generates HDF5 files for rawacf and antennas_iq types.

Borealis filetypes
------------------

These are the Borealis filetypes produced by the radar software, from most processed to least processed.

* **rawacf:** The correlated data from the main and interferometer arrays. Produced by Borealis in release mode.

* **bfiq:** The beamformed iq data from both arrays.

* **antennas_iq:** The iq data from every antenna. Produced by Borealis in release mode.

* **rawrf:** The unfiltered, full receive bandwidth data from every antenna. Only produced by Borealis in debug modes.

Post-processed dmap files can be created from the hdf5 rawacf or bfiq files using the `pyDARNio
package <https://github.com/superdarn/pydarnio>`_.

For more information on the data files and the fields stored within them, check the data file
information for the correct Borealis software version.

Borealis current version
------------------------

The Borealis software version can affect the data fields in the file format so be sure to check if
your data is of the most up to date version. The current Borealis software version is v0.6.

..  toctree::
    :maxdepth: 1

    rawacf
    bfiq
    antennas_iq
    rawrf

Previous versions
-----------------

..  toctree::
    :maxdepth: 1

    archive/rawacf-v06
    archive/bfiq-v06
    archive/antennas_iq-v06
    archive/rawrf-v06

..  toctree::
    :maxdepth: 1

    archive/rawacf-v05
    archive/bfiq-v05
    archive/antennas_iq-v05
    archive/rawrf-v05

..  toctree::
    :maxdepth: 1

    archive/rawacf-v04
    archive/bfiq-v04
    archive/antennas_iq-v04
    archive/rawrf-v04

v0.2 and v0.3  follow the v0.4 format.

------------
Reading Data
------------

To read the files in python, we recommend using `h5py <https://docs.h5py.org/en/stable/>`_ package.
If you are looking to generate SuperDARN standard plots, we recommend using the the `pyDARN package
<https://github.com/superdarn/pydarn>`_, which can read Borealis files specifically. After
converting to dmap, standard SuperDARN plots including RTI plots and fan plots can be produced.

-------------------------
Data Storage and Deletion
-------------------------

Borealis file sizes can add up quickly to fill all available hard drive space, especially if
antennas_iq and/or bfiq data types are being generated. However, it is convenient and recommended to
keep a backlog of lower level data products such as antennas_iq for a period of time. These files
are useful for debugging hardware issues and reproducing RAWACF files.


File Rotation
-------------

In order to prevent system failure due to hard drives filling up, a method for deleting the oldest
data files is employed for SuperDARN Canada radars. This is referred to as *rotating* the files.

A utility script is scheduled via cron to check the filesystem that Borealis files are written to.
If the filesystem usage is too high, it searches for and deletes the oldest files in a loop until
the filesystem usage goes below the threshold. See the SuperDARN Canada `data flow repository
<https://github.com/SuperDARNCanada/data_flow>`_ for more information.
