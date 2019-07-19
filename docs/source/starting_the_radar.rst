******************
Starting the Radar
******************

===============
Manual Start-up
===============

To more easily start the radar, there is a script called `steamed_hams.sh`. The name of this script is a goofy reference to a scene in an episode of The Simpsons in which Principal Skinner claims there is an aurora happening in his house. The script takes two arguments and can be invoked as follows:

    * $BOREALISPATH/steamed_hams.sh experiment_name code_environment

The experiment name must match to an experiment in the experiment folder, and does not include the .py extension. The code environment is the type of compilation environment that was compiled using scons such as release, debug, etc.

The script will boot all the radar processes in a detached `screen` window that runs in the background. This window can be reattached in any terminal window locally or over ssh to track any outputs if needed.

==================
Automated Start-up
==================

The scheduling Python script, `remote_server.py`, is responsible for automating the control of the radar to follow the schedule. This script should be added the control computer bootup scripts so that it generate a new set of scheduled commands.

