.. Borealis documentation master file, created by
   sphinx-quickstart on Fri Jan  5 21:28:44 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

Welcome to Borealis's documentation!
====================================

Introduction
____________
Borealis is a newly developed digital radar system by the engineering team at SuperDARN Canada.
It is a substantial upgrade to existing SuperDARN systems. Features of Borealis include:

- Several new experimental capabilities (see :ref:`new-experiments`)
- Improved diagnostics and telemetry (see :ref:`monitoring` and the
  :ref:`realtime package <realtime-package>`)
- Flexible and easy-to-implement experiments (see :ref:`experiments`)
- Direct sampling of each antenna receive path in the standard SuperDARN array, resulting in flexible
  post-processing of data if the samples from each antenna are stored (see :ref:`antennas_iq`)

A paper has been published for the Borealis system and can be found
`here <https://doi.org/10.1029/2022RS007591>`_.

This documentation attempts to capture all information required for a new user to Borealis to start
from nothing and eventually have a Borealis system running. The information includes:

- What :ref:`parts` to buy
- How to interface with :ref:`existing transmitters <transmitter-interface>` and how the
  :ref:`Canadian radars do it <txio-board>`
- What :ref:`hardware` and :ref:`software` modifications are required
- What :ref:`options <config-options>` there are within the Borealis system and how to configure them
- How to set up :ref:`tests <lab-testing>` in order to verify a working system
- How to write your own custom :ref:`experiments <building-experiments>`
- How data produced by Borealis :ref:`map <rawacf-sdarn-mapping>` to formats
  agreed upon by the SuperDARN community
- How to simulate a SuperDARN antenna array with :ref:`NEC` using our custom python script
- And of course no documentation is complete without a list of :ref:`common issues <failure-modes>`

This documentation is always being updated and refined, so please check back regularly for updates.
Any comments, questions, and/or suggestions can be sent to the
`SuperDARN Canada team <https://superdarn.ca/contact>`_, we welcome any and all feedback, as that
has helped to make the Borealis system as successful as it is today!


Version 0.7
-----------
Several exciting improvements have been made to Borealis in the most recent release, improving the flexibility and
reliability of the system. The changes include:

* Logging: ``structlog`` is now used for consistent log formatting, with pretty formatting in the console and JSON
  formatting saved to logfiles. Log forwarding is also supported using the graypy extended log format (GELF). Separate
  log levels can be used for each type of log handler (console, logfile, or aggregator), specified in the config file.
* Config file format: The config file format has been updated to be more intuitive, particularly for configuring the
  connections to N200s. Each N200 field in the config file now supports an ``addr`` field and three channel fields.
  These channel fields map to the one transmit and two receive channels on the front of each N200. The value of each
  field is a physical antenna that the channel is connected to. This format removes any redundancy or edge case
  configurations that previously were not supported. See :ref:`options <config-options>` for the full config file
  format.
* Directory structure: The code base has been refactored. The source code for running the radar is in ``src/``,
  scripts for starting/stopping the radar, installing dependencies, etc. are in ``scripts/``, the scheduler code is
  in ``scheduler/``, config files are in ``config/``, and tests are in ``tests/``.
* The ``DecimationScheme`` object for filtering and downsampling raw voltage samples can now be defined per slice.
  Slices that are CONCURRENT interfaced must share the same ``DecimationScheme``.
* Complex phases of transmit antennas now stored in data files.
* Experiments may now specify a function for calculating the complex phase for each antenna for beamforming received
  data.
* Negative CPID is only used if the ``--embargo`` flag is passed to ``steamed_hams.py``. This can be configured in the
  schedule as needed.
* In multi-beam averaging periods, all beams are served by the ``realtime`` module.
* Enabled tuning of the N200 center frequencies for TX and RX during an experiment. Tuning occurs at the start of an
  averaging period, so all slices that are SEQUENCE or CONCURRENT interfaced must share center frequencies.
* Created a Jupyter notebook for testing out new DecimationScheme objects and visualizing their performance.
* Added more options for array configuration in the NEC generation script.
* Created a daemon for automating restarts of the radar, with optional Slack webhook integration for automated alerts
  if the radar has restarted consecutive times.
* Removed the email functionality of the scheduler.
* Added Slack webhook integration to notify when a schedule file has been synced to a site computer.
* Updated the scheduler to verify that all future experiments are valid, i.e. the experiment files exist and the
  fields of each slice pass the internal checks of Borealis
* Made the ``realtime`` process optional. It can be disabled by passing ``--realtime-off`` to ``steamed_hams.py``.

Limitations
^^^^^^^^^^^
- Borealis does not implement clear frequency search before transmitting. Only fixed frequencies
  are used.
- The 5MHz transmit and receive bandwidths are effectively only 3.5MHz wide, as the transmit
  waveform near the edges of the band seem to have issues. This requires further investigation.


Roadmap
^^^^^^^
In the next release, we plan to implement:

- The option to have pulse compression phase codes on the transmit pulses
- The option to use a clear frequency search
- Remove the dependency on ``protobuf``
- Serve metadata for experiments that don't produce RAWACF data
- Continual improvements to the code-base and documentation


..  toctree::
    :maxdepth: 2
    :hidden:
    :glob:

    /source/system_specifications
    /source/system_setup
    /source/config_options
    /source/starting_the_radar
    /source/lab_testing
    /source/transmitter_interface
    /source/txio_board
    /source/scheduling
    /source/monitoring
    /source/building_an_experiment
    /source/new_experiments
    /source/borealis_processes
    /source/borealis_data
    /source/postprocessing
    /source/tools
    /source/failure_modes
    /source/development
    /source/glossary

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
