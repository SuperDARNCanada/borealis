===============================
Starting and Stopping the Radar
===============================

---------------
Manual Start-up
---------------

To more easily start the radar, there is a script called ``steamed_hams.py``. The name of this
script is a goofy reference to a scene in an episode of The Simpsons in which Principal Skinner
claims there is an aurora happening in his house. The script takes two arguments and can be invoked
as follows: ::

    $BOREALISPATH/scripts/steamed_hams.py experiment_name code_environment scheduling_mode

An example invocation to run ``twofsound`` in ``release`` mode would be: ::

    /home/radar/borealis/scripts/steamed_hams.py twofsound release

Another example invocation running ``normalscan`` in ``debug`` mode: ::

    /home/radar/borealis/scripts/steamed_hams.py normalscan debug

Another example invocation running epopsound in debug mode during special time would be: ::

    /home/radar/borealis/scripts/steamed_hams.py epopsound debug special

The experiment name must match to an experiment in the ``src/borealis_experiments`` folder, and does
not include the ``.py`` extension. The code environment is the type of compilation environment that
was compiled using ``scons`` such as ``release``, ``debug``, etc. **NOTE** This script will kill the
Borealis software if it is currently running, before it starts it anew. The scheduling mode is one
of ``common``, ``special``, or ``discretionary`` depending upon the DARN-SWG schedule (see the
scheduling working group page `here http://superdarn.thayer.dartmouth.edu/WG-sched/charter.html`_)

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
Automated restarts
------------------

Occasionally, the Borealis software stops due to some software or computer issue.
In order to automatically restart the radar software when this occurs, and to avoid lengthy
downtimes, a script called restart_borealis.py was written. This script is called via crontab
periodically (every 10 minutes).
The script checks for a config file, finds out where the data directory is from the configuration
file, and then looks for the data file that is currently being written to.
If the file was written to recently, it does nothing, if a certain threshold of time has
passed since the file was written to, then it assumes that the radar has stopped running
properly, and attempts to restart it.

The crontab entry is shown below: ::

    */10 * * * * . $HOME/.profile; /usr/bin/python3 /home/radar/borealis/scripts/restart_borealis.py >> /home/radar/borealis/restart_log.txt 2>&1

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