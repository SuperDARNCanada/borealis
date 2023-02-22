.. Borealis documentation master file, created by
   sphinx-quickstart on Fri Jan  5 21:28:44 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Borealis's documentation!
====================================

Introduction
____________
Borealis is a newly developed digital radar system by the engineering team at SuperDARN Canada.
It is a substantial upgrade to existing SuperDARN systems. Features of Borealis include:

- Several new experimental capabilities (see :ref:`new-experiments`)
- Improved diagnostics and telemetry (see :ref:`monitoring`)
- Flexible and easy-to-implement experiments (see :ref:`experiments`)
- Direct sampling of each antenna receive path in the standard SuperDARN array, resulting in flexible
  post-processing of data if the samples from each antenna are stored (see :ref:`antennas_iq`)

A paper has been published for the Borealis system and can be found
`here <https://doi.org/10.1029/2022RS007591>`_

This documentation attempts to capture all information required for a new user to Borealis to start
from nothing and eventually have a Borealis system running. The information includes:

- What :ref:`parts` to buy
- What :ref:`hardware<Hardware>` and :ref:`software<Software>` modifications are required parts
- What :ref:`options<config options>` there are within the Borealis system and how to configure them
- How to set up :ref:`tests<Lab Testing>` in order to verify a working system
- How to write your own custom :ref:`experiments<Building an Experiment>`
- How data produced by Borealis :ref:`map<rawacf to rawacf SDARN (DMap) Conversion>` to formats
  agreed upon by the SuperDARN community
- How to simulate a SuperDARN antenna array with :ref:`NEC` using our custom python script
- :ref:`Common issues<Common Failure Modes>`

This documentation is always being updated and refined, so please check back regularly for updates.
Any comments, questions, and/or suggestions can be sent to the
`SuperDARN Canada team<https://superdarn.ca/contact>`_, we welcome any and all feedback, as that
has helped to make the Borealis system as successful as it is today!


Limitations
-----------
Current limitations as of February, 2023:

- Borealis does not implement clear frequency search before transmitting. Only fixed frequencies
  are used
- The 5MHz transmit and receive bandwidths are effectively only 3.5MHz wide, as the transmit
  waveform near the edges of the band seem to have issues. This requires further investigation
- There is currently no ability to tune centre frequencies of the USRP units in real-time
- All slices of an experiment must have the same decimation scheme


Roadmap
-------
In the near future, we plan to implement:

- The option to have pulse compression using 13-bit Barker codes on the transmit pulses
- The option to use a clear frequency search
- Continual improvements to the code-base and documentation


..  toctree::
    :maxdepth: 2
    :glob:

    /source/system_specifications
    /source/system_setup
    /source/config_options
    /source/starting_the_radar
    /source/lab_testing
    /source/transmitter_interface
    /source/scheduling
    /source/monitoring
    /source/building_an_experiment
    /source/new_experiments
    /source/borealis_processes
    /source/borealis_data
    /source/tools
    /source/failure_modes
    /source/glossary

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
