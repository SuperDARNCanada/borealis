*******************
Borealis Monitoring
*******************

The monitoring system implemented for Borealis is a custom configured installation of Nagios Core, working with NRPE. Nagios monitoring behaves according to objects defined in configuration files, all of which have copies in the monitoring directory of Borealis.

Nagios
------
Nagios core runs as a service under apache2. It is easy to install, but a little tricky to configure for specific purposes. The program executes external plugins that obtain information from the system, and then displays the output on locally hosted webpage. Locally, where and which plugins are executed is determined by host and service objects specified in configuration files. This is also done with monitoring on remote machines, with one exception. The remote server runs plugins using an a service called NRPE (Nagios Remote Plugin Executor). This process runs on port 566 by default, and sends plugin output over the network to the Nagios service running on the central host. The central host accepts this output through a plugin called check_nrpe, usage specified in the commands.cfg config file. This remote host output is then displayed normally alongside the local services.


Installation
------------
