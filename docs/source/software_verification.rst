.. _software-verification:

=====================
Software Verification
=====================

.. _Radar Simulator:

Radar Simulator
---------------

Usage::

    python3 tests/simulators/steamed_sham.py normalscan release discretionary; screen -r borealis

The most comprehensive test of a custom experiment is to run the radar with it. Without a lab setup, this can be costly
in terms of radar time. As such, a simulator script ``tests/simulators/steamed_sham.py`` has been created to run
the radar software without USRPs, utilizing a USRP driver simulator script ``tests/simulators/driver_sim.py``.

Running the script will open up the typical radar screens, with the USRP driver window replaced by the ``driver_sim``
process. Since there are no USRPs to configure, the radar begins operating much more quickly. All other radar modules
(e.g. ``radar_control``, ``data_write``, etc) are running the exact code that normally operates the radar, thus
providing a comprehensive test of the experiment functionality. Note that ``driver_sim.py`` generates noise for the RX
channels, so the actual data written to file and served by the ``realtime`` process is nonsensical. However, all
metadata fields should be accurate, and thus can be inspected to verify the experiment ran as expected.

.. _Scheduler Testing:

Scheduler
---------

Usage::

    python3 tests/scheduler/test_scheduler.py

Verifies correct operation of the scheduling functions and classes.

.. _Filter Testing:

Filter Scheme
-------------

A Jupyter notebook is located at ``tests/dsp_testing/filters.ipynb``.
This notebook describes in detail the default :class:`DecimationScheme` used by Borealis, the helper functions in
``decimation_scheme.py`` for creating a digital filter, and creates an alternative filter with comparison
to the default. This notebook also benchmarks the performance of the filter schemes on the GPU, both in runtime
and in memory usage. Finally, the ramp-up and ramp-down of transmitted pulses is looked at, for characterization
of the expected transmission spectrum.

This notebook is intended to make it easy to design and prototype new filtering schemes, which is useful for
experiments which, for example:

* Have a non-standard pulse length (the default is 300 microseconds)
* Use range gates of differing size (the default is 45 km)
* Use pulse compression (this increases the transmission bandwidth)
* Are listening experiments (e.g. to measure the frequency spectrum)
* Use a different receiver bandwidth (i.e. not the default 5 MHz)

In any of these circumstances, it is important to design a filter which works for the situation at hand.

.. _Config Testing:

Config File
-----------

Usage::

    config_testing.py [-h] [configs ...]

    positional arguments:
      configs     List of config files to check. If none specified, will run unit
                  tests.

    options:
      -h, --help  show this help message and exit

A Python ``unittest`` script for verifying config files is located at ``tests/config_files/config_testing.py``.
Note that this script only verifies that the config file is valid, and makes no guarantees about the compatibility of
the config file with custom experiment classes (e.g. running with less than 16 TX antennas is not supported for the
``full_fov`` experiment).

.. _Realtime Testing:

Realtime Data Simulator
-----------------------

Usage::

    python3 tests/simulators/realtime/realtime_sim.py

A simulator script is available at ``tests/simulators/realtime/realtime_sim.py``. This script calls the
``realtime_server()`` function of the realtime module, and tests sending a rawacf record to it repeatedly.
The script logs each record that it sends, and each fitacf response that it receives.
