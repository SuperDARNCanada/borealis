Realtime Package
================

The realtime package contains the utilities that check data files for corruption during operation
and serving the data to external sockets.

This module also handles the removal of temporary files generated during radar operation. Temporary
files are generated every sequence and copied into their respective two hour file. The path to the
temporary file is passed from data_write to realtime, which realtime uses to check the file
integrity and serve the data.

This module is required to operate the radar. Without it, the system will slow as the number of
temporary files increases.
