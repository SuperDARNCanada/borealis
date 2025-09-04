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

Version 1.0
-----------
The latest release of Borealis includes several major software changes. These include:

* Clear frequency search: This new capability is highly configurable, both in operation and analysis. The frequency
  spectrum measured in the search is saved in the HDF5 files, presenting a more detailed look at the noise/interference
  environment than was previously possible.
* HDF5 file structure: gone are ``site`` and ``array`` structured files. HDF5 files are now structured similarly to
  the old ``site`` structured files, and can be converted in-memory to ``array`` structured data using pyDARNio.
  pyDARNio also supports loading in both structures as ``xarray`` DataSets, easing data exploration for new users.
  In general, the HDF5 files contain much more metadata.
* Protobuf: this dependency has been removed. Large arrays that were previously transferred using protobuf are now
  put in shared memory, with the shared memory address shared instead. A simple bespoke message protocol is used for
  all interprocess communication.
* Pydantic: this dependency has been version bumped to v2.
* Testing: simulator scripts were created for testing the entire system without N200 communication, and for isolated
  testing of the realtime module.
* Default ``DecimationScheme``: a new default was created, consisting of two stages with larger downsampling factors
  between each. This scheme uses significantly less GPU memory and runs significantly faster.
* ``apcupsd`` scripts: these new scripts handle stopping the radar when a power outage occurs, and restarting the radar
  when power is restored.
* Frequency bug: operating at frequencies that are not a multiple of 10 kHz would produce garbage rawacfs in prior
  versions. This bug was identified and fixed. Affected rawacfs can be fixed if the ``antennas_iq`` is still present.
* Config files: new ``antennas`` field created, with the relative locations of each physical antenna. These values are
  used for beamforming. Also new ``rawacf_format`` field, for specifying whether ``rawacf`` fields should be written in
  HDF5 or DMAP format.
* ``txdata``: this data product is no longer supported.
* Pulse sequence timing: the first pulse in each sequence now will always start on a millisecond boundary.
* RX/TX center frequencies: these can be automatically determined based on the slices of an experiment.
* Scheduling daemon: for running ``local_scd_server.py`` persistently.
* Code style: ``ruff`` tool used for linting/formatting.

Limitations
^^^^^^^^^^^
- The 5MHz transmit and receive bandwidths are effectively only 3.5MHz wide, as the transmit
  waveform near the edges of the band seem to have issues. This requires further investigation.

Roadmap
^^^^^^^
In the next release, we plan to implement:

- The option to have pulse compression phase codes on the transmit pulses
- Modify the clear frequency search implementation to use dead times between pulse sequences
- Allow slices to be grouped in files (e.g. all slices except 0 stored together for normalsound).
- Serve metadata for experiments that don't produce RAWACF data
- Serve DMAP data directly from the realtime module, instead of JSON
- Continual improvements to the codebase and documentation


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
