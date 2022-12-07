**********
Scheduling
**********

Borealis scheduling is made of several components to help automate and reduce overhead in
scheduling. The idea here is to have a script that runs locally at the institution which generates
new schedules, a cloud syncing service to automatically upload the new schedules to the radar sites,
and then a remote script on site that converts the schedules to actual radar commands.

The local script will monitor the Scheduling Working Group (SWG) web link for new uploads and then
grab them if there is anything new. At the time of writing, these files are hosted at
`<https://github.com/SuperDARN/schedules>`_. This automated script will then parse the lines from
the file and convert them to schedule file (SCD) commands.

The schedule files need to be synced to the radar sites. It is recommended to set up a directory
which is cloud shared using a service such as Nextcloud or Owncloud. The SCD files that the local
script adds to should all be in this directory so that syncing is all automated.

The remote script will check for changes to any synced files and then generate ``at`` command
arguments for Borealis experiments to run. This allows us to utilize scheduling utilities already
available in Linux.

These scripts are configured with logging and email capability so that maintainers can track if
scheduling is successful. There is also a utility script called ``schedule_modifier.py`` that should
be used to add or remove lines from the schedule so that no errors are made in the schedule file. It
is not recommended to manually modify any schedule files.

Here is a simple diagram for how scheduling works. It starts with the DSWG repository, which is
accessed via a local server, which then uses unison to sync with all Borealis radars.

.. image:: img/scheduling_diagram.png
    :scale: 100%%
    :alt:   Simple block diagram of scheduling setup
    :align: center

Here are the steps to configure scheduling:

1. Configure a local institution server to build schedules.

    - Configure a cloud/network syncing service such as unison or NFS. Configure this service to
      share a directory where schedules and logs are to be stored.
    - Git clone a copy of Borealis.
    - Edit the ``local_scd_server.py`` with the correct experiments and radars belonging to your
      institution.
    - Configure a system service or reboot ``cron`` task to run the python3 script
      ``local_scd_server.py`` at boot. This script requires the argument ``--scd-dir`` for the
      schedules directory as well as ``--emails-filepath`` which should be a text file of emails on
      each line where scheduling status will be sent.
    - The ``local_scd_server.py`` script has an option for running manually the first time to
      properly configure the scheduling directory with the schedules for the latest files available.
    - Example: ::
        
        python3 ./local_scd_server.py --first-run --scd-dir=/data/borealis_schedules --emails-filepath=/data/borealis_schedules/emails.txt


2. Configure the Borealis computer.
        
    - unison will execute on the remote and connect to this machine to sync.
    - Schedule a reboot task via ``cron`` to run the ``start_radar.sh`` helper script in order to
      run the radar according the radar schedule.
    - Enable and start ``atq`` service.

**Scheduler Code**

.. toctree::
   :glob:
   :maxdepth: 1

   code_docs/scheduler_code.rst