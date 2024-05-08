.. _monitoring:

===================
Borealis Monitoring
===================

The monitoring system implemented for Borealis is a custom configured installation of Nagios Core,
working with NRPE. Nagios monitoring behaves according to objects defined in configuration files,
all of which have copies in SuperDARN Canada's Nagios repository.

------
Nagios
------
Nagios core runs as a service under apache2. It is easy to install, but a little tricky to configure
for specific purposes. The program executes external plugins that obtain information from the
system, and then displays the output on locally hosted webpage. Locally, where and which plugins are
executed is determined by host and service objects specified in configuration files. This is also
done with monitoring on remote machines, with one exception.

The remote server runs plugins using an a service called NRPE (Nagios Remote Plugin Executor). This
process runs on TCP port 5666 by default, and sends plugin output over the network to the Nagios service
running on the central host. The central host accepts this output through a plugin called
check_nrpe, usage specified in the commands.cfg config file. This remote host output is then
displayed normally alongside the local services.

In our configuration, remote hosts send information on services continuously, allowing connections
from hosts specified in their nrpe.cfg file. To operate properly, both the hostname of the remote
host, and that of the central Nagios host, must be included on this line.

The last key difference between NRPE and Nagios Core is that commands to be executed on the remote
host are defined in that host's nrpe.cfg file. Whereas commands executed by Nagios Core are defined
in the commands.cfg by default.

Installation
------------
Detailed instructions for installing Nagios Core on several operating systems can be found on
Nagios' website_.

.. _website: https://assets.nagios.com/downloads/nagioscore/docs/nagioscore/4/en/quickstart.html

Installation of NRPE is similarly simple. We maintain a private repository with our Nagios configuration.
Contact us if you would like to set up Nagios for your own system, we would be happy to share our
experience.

-------------------
Downtime Monitoring
-------------------
A useful metric for measuring the reliability of the system is to quantify the amount of downtime (or uptime), that is,
the percentage of time that the radar is non-operating (or operating). A companion repository,
`borealis-data-utils <https://github.com/SuperDARNCanada/borealis-data-utils>`_, contains a script called
`borealis_gaps.py <https://github.com/SuperDARNCanada/borealis-data-utils/blob/main/borealis_gaps.py>`_ which reads
through a directory of Borealis HDF5 files and reports all downtimes in a given date range. The script takes several
command line options, allowing the user to specify the date range, type of file to search for, minimum downtime duration
to report, number of processes to use, and output format of the report. Follow the links above for more details.
