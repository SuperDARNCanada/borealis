===============================
Starting and Stopping the Radar
===============================

---------------
Manual Start-up
---------------

To more easily start the radar, there is a script called ``steamed_hams.sh``. The name of this
script is a goofy reference to a scene in an episode of The Simpsons in which Principal Skinner
claims there is an aurora happening in his house. The script takes two arguments and can be invoked
as follows: ::

    $BOREALISPATH/scripts/steamed_hams.sh experiment_name code_environment

An example invocation to run ``twofsound`` in ``release`` mode would be: ::

    /home/radar/borealis/scripts/steamed_hams.sh twofsound release

Another example invocation running ``normalscan`` in ``debug`` mode: ::

    /home/radar/borealis/scripts/steamed_hams.sh normalscan debug

The experiment name must match to an experiment in the ``src/borealis_experiments`` folder, and does
not include the ``.py`` extension. The code environment is the type of compilation environment that
was compiled using ``scons`` such as ``release``, ``debug``, etc. **NOTE** This script will kill the
Borealis software if it is currently running, before it starts it anew.

The script will boot all the radar processes in a detached ``screen`` window that runs in the
background. This window can be reattached in any terminal window locally or over ssh (``screen -r``)
to track any outputs if needed.

If starting the radar in normal operation according to the schedule, there is a helper script called
``start_radar.sh``.

------------------
Automated Start-up
------------------

In order to start the radar automatically, the script ``start_radar.sh`` should be added to a
startup script of the Borealis computer. It can also be called manually by the non-root user
(typically ``radar``). 

The scheduling Python script, ``remote_server.py``, is responsible for automating the control of the
radar to follow the schedule, and is started via the ``start_radar.sh`` script (shown :ref:`below
<start_radar-sh>`) with the appropriate arguments.

This script should be added to the control computer boot-up scripts so that it generates a new set
of scheduled commands.

------------------
Stopping the Radar
------------------

There are several ways to stop the Borealis radar. They are ranked here from most acceptable to
last-resort:

#. Run the script ``stop_radar.sh`` from the Borealis ``scripts/`` directory. This script kills the
   scheduling server, removes all entries from the schedule and kills the screen session running the
   Borealis software modules. ``stop_radar.sh`` is shown :ref:`below <stop_radar-sh>`.

#. While viewing the screen session running the Borealis software modules, type ``ctrl-A, ctrl-\\``.
   This will kill the screen session and all software modules running within it.

#. Restart the Borealis computer. **NOTE** In a normal circumstance, the Borealis software will
   start back up again once the computer reboots.

#. Shut down the Borealis computer.

-------
Scripts
-------

..  literalinclude:: ../../scripts/start_radar.sh
    :language: bash
    :linenos:
    :caption: start_radar.sh
    :name: start_radar-sh

..  literalinclude:: ../../scripts/stop_radar.sh
    :language: bash
    :linenos:
    :caption: stop_radar.sh
    :name: stop_radar-sh