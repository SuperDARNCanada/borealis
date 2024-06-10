==========
Scheduling
==========

Borealis scheduling is made of several components to help automate and reduce overhead.
The idea here is to have a script that runs locally at the institution which generates
new schedules, which are then synchronized automatically to the radar sites,
and then a remote script on site that converts the schedules to actual radar commands.

The local script will monitor the Scheduling Working Group (SWG) web link for new uploads and then
grab them if there is anything new. At the time of writing, these files are hosted at
`<https://github.com/SuperDARN/schedules>`_. This automated script will then parse the lines from
the file and convert them to schedule file (SCD) commands.

The schedule files need to be synced to the radar sites. The SCD files that the local script adds to
should all be in this directory so that syncing is all automated. This syncing is currently done via
a daemon process (``scheduler_sync.daemon``) that continually watches the local files for changes
using inotify, then rsyncs the changed files to each site. If a schedule fails to sync, an alert is
sent to our group's Slack workspace to notify us. For more information on integrating Slack alerts,
see `here <https://www.howtogeek.com/devops/how-to-send-a-message-to-slack-from-a-bash-script/>`__.

The remote script (``remote_server.py``) will check for changes to any synced files and then generate
``atq`` command arguments for Borealis experiments to run. This allows us to utilize scheduling
utilities already available in Linux.

These scripts are configured with logging capability so that maintainers can track if
scheduling is successful. There is also a utility script called ``schedule_modifier.py`` that should
be used to add or remove lines from the schedule so that no errors are made in the schedule file. It
is not recommended to manually modify any schedule files.

Here is a simple diagram for how scheduling works. It starts with the DSWG repository, which is
accessed via a local server, which then uses the scheduler sync daemon to sync with all Borealis
radars.

.. image:: img/scheduling_diagram.png
    :scale: 100%%
    :alt:   Simple block diagram of scheduling setup
    :align: center

Here are the steps to configure scheduling:

1. Configure a local institution server to build schedules.

    - Git clone a copy of Borealis.
    - Configure scheduler sync daemon script to sync to the various Borealis radar computers.
    - Edit the ``local_scd_server.py`` with the correct experiments and radars belonging to your
      institution.
    - Configure environment variables necessary to run the scheduler. Create a file ``{HOME}/.scheduler``
      and add the lines: ::

        declare -A RADAR_PORTS=(["AAA_BBBB"]=xxxxx ["CCC_DDDD"]=yyyy)
        SCHEDULER_DEST="username@host"

      Here "AAA" and "CCC" are three-letter site IDs, in all caps, and "BBBB" and "DDDD" are computer names,
      such as BORE or MAIN, also in all caps. xxxxx and yyyyy are port numbers needed to connect to the site computers.
      Two ports are shown, but if you have more or less site computers you can modify the entries of the array as
      needed. SCHEDULER_DEST defines the username and hostname for the connection to the site computers, assuming all
      site computers managed have the same username and hostname. This can be achieved if using autossh for persistent
      ssh connections between the computer and site computers.
    - If using Slack as a platform, you can add the environment variable ``SLACK_WEBHOOK_[RADAR_ID]`` to
      ``{HOME}/.profile`` to get notifications whenever the schedule is changed on the server computer and synced
      with a site computer. For example, we define: ::

        export SLACK_WEBHOOK_SAS=https://hooks.slack.com/services/{specific url here}

      which sends a message to a specific channel of our Slack workspace whenever the scheduler makes changes to
      ``sas.scd``.
    - Configure a system service or reboot ``cron`` task to run the python3 script
      ``local_scd_server.py`` at boot. This script requires the argument ``--scd-dir`` for the
      schedules directory.
    - The ``local_scd_server.py`` script has an option for running manually the first time to
      properly configure the scheduling directory with the schedules for the latest files available.
    - Example: ::

        python3 ./local_scd_server.py --first-run --scd-dir=/data/borealis_schedules


2. Configure the Borealis computer.

    - Schedule a reboot task via ``cron`` to run the ``start_radar.sh`` helper script in order to
      run the radar according the radar schedule.
    - Enable and start ``atq`` service.

**Scheduler Code**

.. toctree::
   :glob:
   :maxdepth: 1

   scheduler_code.rst
