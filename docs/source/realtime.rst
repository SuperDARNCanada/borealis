================
Realtime Package
================

The realtime package contains the utilities that check data files for corruption during operation and serving the data
to external sockets.

This module also handles the removal of temporary files generated during radar operation. Temporary files are generated
every averaging period and copied into their respective two hour file. The path to the temporary file is passed from
data_write to realtime, which realtime uses to check the file integrity, perform additional processing, and serve the
resulting data. Once this is complete, the realtime module deletes the temporary file.

Additional data processing is done using a Python package called ``backscatter``, which is an implementation of the
standard FITACF3 algorithm for analyzing SuperDARN data. More details about this package can be found at its GitHub
`homepage <https://github.com/SuperDARNCanada/backscatter>`_.

This module is required to operate the radar. Without it, the system will slow as a backlog of temporary files grows
and the data write module tries to send the file names over a socket but there is nothing on the other end of the
socket to receive these names.
