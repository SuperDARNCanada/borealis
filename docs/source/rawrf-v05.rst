==========
rawrf v0.5
==========

The pyDARNio format class for this format is BorealisRawrfv0_5 found in the `borealis_formats <https://github.com/SuperDARN/pyDARNio/blob/master/pydarnio/borealis/borealis_formats.py>`_.

Borealis software version 0.5 is out of date, see the current format of the rawrf files `here <https://borealis.readthedocs.io/en/latest/borealis_data.html#borealis-current-version>`_.

This format is intended to hold high bandwidth, non-filtered raw data from every antenna.

This format is only produced in a site-style, record by record format and is only available to be produced on request. Please note that this format
can cause radar operating delays and may reduce number of averages in an integration, for example. 

----------------
rawrf site files
----------------

Site files are produced by the Borealis code package and have the data in a record by record style format. In site files, the hdf5 group names (ie record names) are given as the timestamp in ms past epoch of the first sequence or sampling period recorded in the record. 

The naming convention of the rawrf site-structured files are:

[YYYYmmDD].[HHMM].[SS].[station_id].rawrf.hdf5.site

For example: 20191105.1400.02.sas.rawrf.hdf5.site

This is the file that began writing at 14:00:02 UT on November 5 2019 at the Saskatoon site, and it provides data the experiment that ran at that time.
Since rawrf is not filtered, this data does not need a slice identifier because it contains all the samples being taken at that time. Some familiarity 
with the experiment may be necessary to understand the data, or some access to the other file types produced concurrently. This is primarily a debug 
format for engineering purposes and should only be produced for special cases.

These files are often bzipped after they are produced.

The file fields under the record name in rawrf site files are:

+----------------------------------+---------------------------------------------+
| | **Field name**                 | **description**                             |
| | *type*                         |                                             |  
+==================================+=============================================+
| | **blanked_samples**            | | Samples that should be blanked because    |
| | *uint32*                       | | they occurred during transmission times,  |
| | [number of blanked             | | given by sample number (index into        |
| | samples]                       | | decimated data). Can differ from the      |
| |                                | | pulses array due to multiple slices in a  |
| |                                | | single sequence.                          |
+----------------------------------+---------------------------------------------+
| | **borealis_git_hash**          | | Identifies the version of Borealis that   | 
| | *unicode*                      | | made this data. Contains git commit hash  | 
| |                                | | characters. Typically begins with the     | 
| |                                | | latest git tag of the software.           |
+----------------------------------+---------------------------------------------+
| | **data**                       | | A contiguous set of samples (complex      | 
| | *[complex64, ]*                | | float) at given sample rate. Needs to be  | 
| |                                | | reshaped by data_dimensions to be         | 
| |                                | | correctly read.                           |
+----------------------------------+---------------------------------------------+
| | **data_descriptors**           | | Denotes what each data dimension          | 
| | *[unicode, ]*                  | | represents. = ‘num_sequences’,            |
| |                                | | ‘num_antennas’, ‘num_samps’ for           |
| |                                | | rawrf                                     |
+----------------------------------+---------------------------------------------+
| | **data_dimensions**            | | The dimensions in which to reshape the    | 
| | *[uint32, ]*                   | | data. Dimensions correspond to            |
| |                                | | data_descriptors.                         |
+----------------------------------+---------------------------------------------+
| | **experiment_comment**         | | Comment provided in experiment about the  |
| | *unicode*                      | | experiment as a whole.                    |
+----------------------------------+---------------------------------------------+
| | **experiment_id**              | | Number used to identify the experiment.   |
| | *int64*                        | |                                           | 
+----------------------------------+---------------------------------------------+
| | **experiment_name**            | | Name of the experiment file.              |
| | *unicode*                      | |                                           | 
+----------------------------------+---------------------------------------------+
| | **int_time**                   | | Integration time in seconds.              |
| | *float32*                      | |                                           | 
+----------------------------------+---------------------------------------------+
| | **intf_antenna_count**         | | Number of interferometer array antennas   |
| | *uint32*                       | |                                           | 
+----------------------------------+---------------------------------------------+
| | **main_antenna_count**         | | Number of main array antennas             |
| | *uint32*                       | |                                           | 
+----------------------------------+---------------------------------------------+
| | **num_samps**                  | | Number of samples in the sampling         |
| | *uint32*                       | | period. Each sequence has its own         |
| |                                | | sampling period. Will also be provided    |
| |                                | | as the last data_dimension value.         |
+----------------------------------+---------------------------------------------+
| | **num_sequences**              | | Number of sampling periods (equivalent to | 
| | *int64*                        | | number sequences transmitted) in the      | 
| |                                | | integration time.                         |
+----------------------------------+---------------------------------------------+
| | **num_slices**                 | | Number of slices used simultaneously in   | 
| | *int64*                        | | this record by the experiment. If more    | 
| |                                | | than 1, data should exist in another file | 
| |                                | | for this time period for the other slice. |
+----------------------------------+---------------------------------------------+
| | **rx_center_freq**             | | Center frequency of the sampled data      | 
| | *float64*                      | | in kHz.                                   |
+----------------------------------+---------------------------------------------+
| | **rx_sample_rate**             | | Sampling rate of the samples in this      | 
| | *float64*                      | | file's data in Hz.                        |
+----------------------------------+---------------------------------------------+
| | **samples_data_type**          | | C data type of the samples, provided for  | 
| | *unicode*                      | | user friendliness. = 'complex float'      |
+----------------------------------+---------------------------------------------+
| | **scan_start_marker**          | | Designates if the record is the first in  | 
| | *bool*                         | | a scan (scan is defined by the            | 
| |                                | | experiment).                              |
+----------------------------------+---------------------------------------------+
| | **scheduling_mode**            | | The mode being run during this time       | 
| | *unicode*                      | | period (ex. 'common', 'special',          |
| |                                | | 'discretionary').                         |
+----------------------------------+---------------------------------------------+
| | **sqn_timestamps**             | | A list of GPS timestamps corresponding to | 
| | *[float64, ]*                  | | the beginning of transmission for each    | 
| |                                | | sampling period in the integration time.  | 
| |                                | | These timestamps come from the USRP       | 
| |                                | | driver and the USRPs are GPS disciplined  | 
| |                                | | and synchronized using the Octoclock.     | 
| |                                | | Provided in milliseconds since epoch.     |
+----------------------------------+---------------------------------------------+
| | **station**                    | | Three-letter radar identifier.            |
| | *unicode*                      | |                                           | 
+----------------------------------+---------------------------------------------+

------------------------
Site/Array Restructuring
------------------------

File restructuring to array files is not done for this format.
