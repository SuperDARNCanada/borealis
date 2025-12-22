.. _scheduling:

Scheduling
==========

Borealis scheduling is made of several components to help automate and reduce overhead.
The idea here is to have a script that runs locally at the institution which generates
new schedules, which are then synchronized automatically to the radar sites,
and then a remote script on site that converts the schedules to actual radar commands. This makes it
easy to modify the radar schedules in one place rather than having to log into each radar to
modify their schedules individually.

The schedule files for Borealis are plain-text files, with each line of the file signifying a distinct
schedule entry. The block below shows a few lines from one of our schedule files::

   20251013 19:40 15 10 full_fov common
   20251013 21:15 15 10 full_fov common
   20251014 18:15 15 10 full_fov common
   20251014 19:50 15 10 full_fov common
   20251014 21:25 15 10 full_fov common
   20251020 00:00 - 0 twofsound discretionary
   20251020 00:00 12960 10 full_fov discretionary
   20251029 00:00 - 0 normalscan common
   20251101 00:00 - 0 normalscan common
   20251105 00:00 - 0 normalsound common
   20251108 00:00 - 0 twofsound common
   20251118 00:00 - 0 twofsound discretionary
   20251118 00:00 12960 10 full_fov discretionary
   20251127 00:00 - 0 normalscan common
   20251201 00:00 - 0 normalscan common
   20251209 00:00 - 0 normalsound common
   20251212 00:00 - 0 twofsound common
   20251216 00:00 - 0 twofsound discretionary
   20251216 00:00 12960 10 full_fov discretionary
   20251225 00:00 - 0 normalscan common

The columns specify the following::
  1. The date that the entry is for, formatted ``YYYYMMDD``.
  2. The time that the entry should start, formatted ``HH:MM``.
  3. The duration that the experiment should run, in minutes. ``-`` means forever (or, until the next equal-priority entry)
  4. The priority, from 0 (lowest) to 20 (highest).
  5. The experiment name (corresponding to the Python file name less the ``.py`` extension, similar to invoking ``steamed_hams.py``).
  6. The scheduling mode (one of ``common``, ``discretionary``, or ``special``. See the `SWG repository <https://github.com/SuperDARN/schedules>`_ for details.)

In the schedule above, the experiments will be scheduled in order of first to last. Some entries are scheduled to begin at the
same time; in this case, the higher-priority experiment takes precedence.

The first script in the chain is called ```central_scd_server.py``. This script should run continually,
and will monitor the Scheduling Working Group (SWG) web link for new uploads and then grab them if there
is anything new. These files are hosted at `<https://github.com/SuperDARN/schedules>`_.
This automated script will then parse the lines from the file and convert them to schedule file entries as shown above.
This script configures default experiments to run based on the schedule (i.e. ``normalscan`` for no-switching time,
``twofsound`` for other common time and discretionary time, etc.) which can be modified to suit your desired defaults.

The schedule files need to be synced to the radar sites. The schedule files that the ``central_scd_server`` script modifies
should all be in one directory so that syncing is easily automated. This syncing is currently done via
a daemon process (``scheduler_sync.daemon``) that continually watches the local files for changes
using inotify, then syncs the changed files to each site. If a schedule fails to sync, an alert is
sent to our group's Slack workspace to notify us. For more information on integrating Slack alerts,
see `here <https://www.howtogeek.com/devops/how-to-send-a-message-to-slack-from-a-bash-script/>`_.

The remote script (``make_atq.py``) will check for changes to any files on the computer running the script
and then generate ``atq`` command arguments for Borealis experiments to run. This allows us to utilize scheduling
utilities already available in Linux.

These scripts are configured with logging capability so that maintainers can track if
scheduling is successful. There is also a `TUI application <https://github.com/SuperDARNCanada/schedule_modifier.git>`_
that should be used to manually add or remove lines from the schedule so that no errors are made in the schedule file. It
is not recommended to modify any schedule files using any other method, as that increases the chances of making a mistake
and causing the radar to not run.

Here is a simple diagram for how scheduling works. It starts with the SWG repository, which is
accessed via a local server, which then uses the scheduler sync daemon to sync with all Borealis
radars.

.. figure:: /source/img/light/scheduling.svg
   :width: 100%
   :alt:   Simple block diagram of scheduling setup
   :align: center
   :class: only-light

.. figure:: /source/img/dark/scheduling.svg
   :width: 100%
   :alt:   Simple block diagram of scheduling setup
   :align: center
   :class: only-dark

Here are the steps to configure scheduling:

1. Configure a central server to build schedules.

    - Git clone a copy of Borealis.
    - Configure scheduler sync daemon script to sync to the various Borealis radar computers.
    - Edit the ``central_scd_server.py`` script with the correct experiments and radars belonging to your
      institution.
    - Configure environment variables necessary to run the scheduler. Create a file ``{HOME}/.scheduler``
      and add the lines::

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
      with a site computer. For example, we define::

        export SLACK_WEBHOOK_SAS=https://hooks.slack.com/services/{specific url here}

      which sends a message to a specific channel of our Slack workspace whenever the scheduler makes changes to
      ``sas.scd``.
    - Configure a system service or reboot ``cron`` task to run the python3 script
      ``central_scd_server.py`` at boot. This script requires the argument ``--scd-dir`` for the
      schedules directory. A system service is preferred as it will restart automatically if it fails for any reason.
    - The ``central_scd_server.py`` script has an option for running manually the first time to
      properly configure the scheduling directory with the schedules for the latest files available.
    - Example::

        python3 ./central_scd_server.py --first-run --scd-dir=/data/borealis_schedules

2. Configure the Borealis computer.

    - Schedule a reboot task via ``cron`` to run the ``start_radar.sh`` helper script in order to
      run the radar according the radar schedule.
    - Enable and start ``atd`` service.
    - Set up the ``make_atq.py`` script to automatically schedule the radar when the .scd file is modified. This can
      be done using system services, requiring two files to be added: one a .path file, and the other a .service file.
      These can be added to ``/etc/systemd/system/``, and examples are included below.

monitor-schedule.path::

   [Unit]
   Description="Monitor ${HOME}/borealis_schedules/${RADAR_ID}.scd for changes"

   [Path]
   User=radar
   EnvironmentFile=%h/.borealis.env
   PathChanged=%h/borealis_schedules/${RADAR_ID}.scd
   Unit=schedule-radar.service

   [Install]
   WantedBy=multi-user.target

schedule-radar.service::

   [Unit]
   Description="Run script to schedule the Borealis radar"

   [Service]
   User=radar
   EnvironmentFile=%h/.borealis.env
   ExecStart=${BOREALISPATH}/borealis_env${PYTHON_VERSION}/bin/python ${BOREALISPATH}/scheduler/make_atq.py %h/borealis_schedules/${RADAR_ID}.scd

   [Install]
   WantedBy=multi-user.target
