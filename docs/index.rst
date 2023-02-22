.. Borealis documentation master file, created by
   sphinx-quickstart on Fri Jan  5 21:28:44 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Borealis's documentation!
====================================

Introduction
____________

Borealis is a newly developed digital radar system by the engineering team at SuperDARN Canada.
It is a substantial upgrade to existing SuperDARN systems, with several new experimental
capabilities (see :ref:`new-experiments`).

A paper has been published for the Borealis system and can be found here: [TODO: DOI]

This documentation attempts to capture all information required for a new user to Borealis to start
from nothing and eventually have a Borealis system running. The information includes:

- What parts to buy
- What hardware and software modifications are required for those parts
- What options there are within the Borealis system and how to configure those options
- How to set up tests in order to verify a working system
- How to monitor a currently running system
- What new experimental capabilities Borealis has made possible
- How to write your own custom experiments
- How to simulate a SuperDARN antenna array with NEC using our custom python script
- Common issues
- and so on...

This documentation is always being updated and refined, so please check back regularly for updates.
Any comments, questions, and/or suggestions can be sent to the SuperDARN Canada team, we welcome
any and all feedback, as that has helped to make the Borealis system as successful as it is today!

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
