*******************
Borealis Monitoring
*******************

The monitoring system implemented for Borealis is a custom configured installation of Nagios Core,
working with NRPE. Nagios monitoring behaves according to objects defined in configuration files,
all of which have copies in SuperDARN Canada's Nagios repository.

Nagios
------
Nagios core runs as a service under apache2. It is easy to install, but a little tricky to configure
for specific purposes. The program executes external plugins that obtain information from the
system, and then displays the output on locally hosted webpage. Locally, where and which plugins are
executed is determined by host and service objects specified in configuration files. This is also
done with monitoring on remote machines, with one exception. 

The remote server runs plugins using an a service called NRPE (Nagios Remote Plugin Executor). This
process runs on port 566 by default, and sends plugin output over the network to the Nagios service
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

After installing, simply replace the configuration files with those found in `this
<https://github.com/SuperDARNCanada/Nagios/tree/main>`_ repository.

Installation of NRPE is similarly simple. Detailed instructions can be found in the NRPE.pdf file
located in the monitoring folder along with our config files.