===============================
Starting and Stopping the Radar
===============================

---------------
Manual Start-up
---------------

To more easily start the radar, there is a script called ``steamed_hams.py``. The name of this
script is a goofy reference to a scene in an episode of The Simpsons in which Principal Skinner
claims there is an aurora happening in his house. The script takes two arguments and can be invoked
as follows::

    $BOREALISPATH/scripts/steamed_hams.py experiment_name code_environment scheduling_mode

An example invocation to run ``twofsound`` in ``release`` mode during ``common`` time would be::

    /home/radar/borealis/scripts/steamed_hams.py twofsound release common

Another example invocation running ``normalscan`` in ``debug`` mode during ``discretionary`` time::

    /home/radar/borealis/scripts/steamed_hams.py normalscan debug discretionary

Another example invocation running epopsound in debug mode during special time would be::

    /home/radar/borealis/scripts/steamed_hams.py epopsound debug special

The experiment name must match to an experiment in the ``src/borealis_experiments`` folder, and does
not include the ``.py`` extension. The code environment is the type of compilation environment that
was compiled using ``scons`` such as ``release``, ``debug``, etc. **NOTE** This script will kill the
Borealis software if it is currently running, before it starts it anew. The scheduling mode is one
of ``common``, ``special``, or ``discretionary`` depending upon the DARN-SWG schedule (see the
scheduling working group page `here <http://superdarn.thayer.dartmouth.edu/WG-sched/charter.html>`_)

The script will boot all the radar processes in a detached ``screen`` window that runs in the
background. This window can be reattached in any terminal window locally or over ssh (``screen -r``)
to track any outputs if needed.

To start the radar without the optional ``realtime`` module, pass the flag ``--realtime-off`` and the
module will not be run. For example::

    /home/radar/borealis/scripts/steamed_hams.py normalscan release discretionary --realtime-off

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
Automated Restarts
------------------

Occasionally, the Borealis software stops due to some software or computer issue. To automatically
restart the radar software when this occurs, and to avoid lengthy downtimes, the scripts
``restart_borealis.daemon`` and ``restart_borealis.py`` were created.

``restart_borealis.py`` finds the directory Borealis writes to and checks the file most recently
written to. If the file hasn't been written to within a specified time period, the script assumes
the radar has stopped running and tries to restart it using ``stop_radar.sh`` and
``start_radar.sh``.

``restart_borealis.daemon`` runs continuously, periodically executing ``restart_borealis.py``. If
the radar is restarted consecutive times, an alert is sent to our group's Slack workspace to notify
us that the radar likely has a problem requiring manual intervention. For more information on
integrating Slack alerts, see `here
<https://www.howtogeek.com/devops/how-to-send-a-message-to-slack-from-a-bash-script/>`__.

To set up the daemon using ``systemd``, add a ``.service`` file within ``/usr/lib/systemd/system/``
(for openSUSE). For example, ::

    [Unit]
    Description=Restart borealis daemon

    [Service]
    User=radar
    ExecStart=/home/radar/borealis/scripts/restart_borealis.daemon
    Restart=always

    [Install]
    WantedBy=multi-user.target

Then, ``enable`` and ``start`` the daemon using the ``systemctl`` commands.

Alternatively, ``restart_borealis.py`` can be run via ``crontab``, as shown below: ::

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
