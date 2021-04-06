================
antennas_iq v0.4
================

The pyDARNio format class for this format is BorealisAntennasIqv0_4 found in the `borealis_formats <https://github.com/SuperDARN/pyDARNio/blob/master/pydarnio/borealis/borealis_formats.py>`_.

Borealis software version 0.4 is out of date, see the current format of the antennas_iq files `here <https://borealis.readthedocs.io/en/latest/borealis_data.html#borealis-current-version>`_. 

The antennas_iq format is intended to hold individual antennas I and Q data. The data is not averaged. 

Both site files and array-restructured files exist for this file type. Both are described below.

-----------------------
antennas_iq array files
-----------------------

Array restructured files are produced after the radar has finished writing a file and contain record data in multi-dimensional arrays so as to avoid repeated values, shorten the read time, and improve human readability. Fields that are unique to the record are written as arrays where the first dimension is equal to the number of records recorded. Other fields that are unique to the slice or experiment (and are therefore repeated for all records) are written only once. 

The group names in these files are the field names themselves, greatly reducing the number of group names in the file when compared to site files and making the file much more human readable.

The naming convention of the antennas_iq array-structured files are:

[YYYYmmDD].[HHMM].[SS].[station_id].[slice_id].antennas_iq.hdf5

For example: 20191105.1400.02.sas.0.antennas_iq.hdf5

This is the file that began writing at 14:00:02 UT on November 5 2019 at the Saskatoon site, and it provides data for slice 0 of the experiment that ran at that time. It has been array restructured because it does not have a .site designation at the end of the filename.

These files are zlib compressed which is native to hdf5 and no decompression is necessary before reading using your hdf5 library. 

The file fields in the antennas_iq array files are:

+-----------------------------------+---------------------------------------------+
| | **FIELD NAME**                  | **description**                             |
| | *type*                          |                                             |
| | [dimensions]                    |                                             |
+===================================+=============================================+
| | **antenna_arrays_order**        | | States what order the data is in and      |
| | *unicode*                       | | describes the data layout for the         |
| | [num_antennas]                  | | num_antennas data dimension               |
| |                                 | | Antennas are recorded main array          |
| |                                 | | ascending and then interferometer array   |
| |                                 | | ascending                                 |
+-----------------------------------+---------------------------------------------+
| | **beam_azms**                   | | A list of the beam azimuths for each beam |
| | *float64*                       | | in degrees off boresite. Note that this   |
| | [num_records x                  | | is padded with zeroes for any record      |
| | max_num_beams]                  | | which has num_beams less than the         |
| |                                 | | max_num_beams. The num_beams field should | 
| |                                 | | be used to read the correct number of     | 
| |                                 | | beams for each record.                    |
+-----------------------------------+---------------------------------------------+
| | **beam_nums**                   | | A list of beam numbers used in this slice |
| | *uint32*                        | | in this record. Note that this is padded  |
| | [num_records x                  | | with zeroes for any record which has      |
| | max_num_beams]                  | | num_beams less than the max_num_beams.    |
| |                                 | | The num_beams field should be used to     |
| |                                 | | read the correct number of beams for each |
| |                                 | | record.                                   |
+-----------------------------------+---------------------------------------------+
| | **borealis_git_hash**           | | Identifies the version of Borealis that   |
| | *unicode*                       | | made this data. Contains git commit hash  |
| |                                 | | characters. Typically begins with the     |
| |                                 | | latest git tag of the software.           |
+-----------------------------------+---------------------------------------------+
| | **data**                        | | A set of samples (complex float) at given |
| | *complex64*                     | | sample rate. Note that records that do not|
| | [num_records x                  | | have num_sequences = max_num_sequences    |
| | num_antennas x                  | | will have padded zeros. The num_sequences |
| | max_num_sequences x             | | array should be used to determine the     |
| | num_samps]                      | | correct number of sequences to read for   |
| |                                 | | the record.                               |
+-----------------------------------+---------------------------------------------+
| | **data_descriptors**            | | Denotes what each data dimension          |
| | *unicode*                       | | represents. = 'num_records',              |
| | [4]                             | | ‘num_antennas’, ‘max_num_sequences’,      |
| |                                 | | ‘num_samps’                               |
+-----------------------------------+---------------------------------------------+
| | **data_normalization_factor**   | | Scale of all the filters used,            |
| | *float32*                       | | multiplied, for a total scale to          | 
| |                                 | | normalize the data by.                    |
+-----------------------------------+---------------------------------------------+
| | **experiment_comment**          | | Comment provided in experiment about the  | 
| | *unicode*                       | | experiment as a whole.                    |
+-----------------------------------+---------------------------------------------+
| | **experiment_id**               | | Number used to identify the experiment.   |
| | *int64*                         | |                                           | 
+-----------------------------------+---------------------------------------------+
| | **experiment_name**             | | Name of the experiment file.              |
| | *unicode*                       | |                                           | 
+-----------------------------------+---------------------------------------------+
| | **freq**                        | | The frequency used for this experiment,   |
| | *uint32*                        | | in kHz. This is the frequency the data    |
| |                                 | | has been filtered to.                     |
+-----------------------------------+---------------------------------------------+
| | **int_time**                    | | Integration time in seconds.              |
| | *float32*                       | |                                           | 
| | [num_records]                   | |                                           | 
+-----------------------------------+---------------------------------------------+
| | **intf_antenna_count**          | | Number of interferometer array antennas   |
| | *uint32*                        | |                                           | 
+-----------------------------------+---------------------------------------------+
| | **main_antenna_count**          | | Number of main array antennas             |
| | *uint32*                        | |                                           | 
+-----------------------------------+---------------------------------------------+
| | **noise_at_freq**               | | Noise at the receive frequency, with      |
| | *float64*                       | | dimension = number of sequences.          |
| | [num_records x                  | | 20191114: not currently implemented and   |
| | max_num_sequences]              | | filled with zeros. Still a TODO. Note     |
| |                                 | | that records that do not have             |
| |                                 | | num_sequences = max_num_sequences will    |
| |                                 | | have padded zeros. The num_sequences      |
| |                                 | | array should be used to determine the     |
| |                                 | | correct number of sequences to read for   |
| |                                 | | the record.                               |
+-----------------------------------+---------------------------------------------+
| | **num_beams**                   | | The number of beams to calculate for each |
| | *uint32*                        | | record.                                   | 
| | [num_records]                   | |                                           | 
+-----------------------------------+---------------------------------------------+
| | **num_samps**                   | | Number of samples in the sampling         |
| | *uint32*                        | | period. Each sequence has its own         |
| |                                 | | sampling period. Will also be provided    |
| |                                 | | as the last data_dimension value.         | 
+-----------------------------------+---------------------------------------------+
| | **num_sequences**               | | Number of sampling periods (equivalent to |
| | *int64*                         | | number sequences transmitted) in the      | 
| | [num_records]                   | | integration time for each record. Allows  | 
| |                                 | | the user to correctly read the data up to |
| |                                 | | the correct number and remove the padded  |
| |                                 | | zeros in the data array.                  |
+-----------------------------------+---------------------------------------------+
| | **num_slices**                  | | Number of slices used simultaneously in   |
| | *int64*                         | | the record by the experiment. If more     |
| | [num_records]                   | | than 1, data should exist in another file |
| |                                 | | for the same time period as that record   |
| |                                 | | for the other slice.                      |
+-----------------------------------+---------------------------------------------+
| | **pulse_phase_offset**          | | For pulse encoding phase, in degrees      |
| | *float32*                       | | offset. Contains one phase offset per     | 
| | [number of pulses]              | | pulse in pulses.                          |
+-----------------------------------+---------------------------------------------+
| | **pulses**                      | | The pulse sequence in units of the        |
| | *uint32*                        | | tau_spacing.                              |
| | [number of pulses]              | |                                           | 
+-----------------------------------+---------------------------------------------+
| | **rx_sample_rate**              | | Sampling rate of the samples in this      |
| | *float64*                       | | file's data in Hz.                        |
+-----------------------------------+---------------------------------------------+
| | **samples_data_type**           | | C data type of the samples, provided for  |
| | *unicode*                       | | user friendliness. = 'complex float'      |
+-----------------------------------+---------------------------------------------+
| | **scan_start_marker**           | | Designates if the record is the first in  | 
| | *bool*                          | | a scan (scan is defined by the            |
| | [num_records]                   | | experiment).                              |
+-----------------------------------+---------------------------------------------+
| | **slice_comment**               | | Additional text comment that describes    |
| | *unicode*                       | | the slice written in this file. The slice |
| |                                 | | number of this file is provided in the    |
| |                                 | | filename.                                 | 
+-----------------------------------+---------------------------------------------+
| | **sqn_timestamps**              | | A list of GPS timestamps corresponding to |
| | *float64*                       | | the beginning of transmission for each    | 
| | [num_records x                  | | sampling period in the integration time.  |
| | max_num_sequences]              | | These timestamps come back from the USRP  | 
| |                                 | | driver and the USRPs are GPS disciplined  |
| |                                 | | and synchronized using the Octoclock.     |
| |                                 | | Provided in milliseconds since epoch.     | 
| |                                 | | Note that records that do not have        | 
| |                                 | | num_sequences = max_num_sequences will    | 
| |                                 | | have padded zeros. The num_sequences      | 
| |                                 | | array should be used to determine the     | 
| |                                 | | correct number of sequences to read for   | 
| |                                 | | the record.                               |
+-----------------------------------+---------------------------------------------+
| | **station**                     | | Three-letter radar identifier.            |
| | *unicode*                       | |                                           | 
+-----------------------------------+---------------------------------------------+
| | **tau_spacing**                 | | The minimum spacing between pulses in     | 
| | *uint32*                        | | microseconds. Spacing between pulses is   | 
| |                                 | | always a multiple of this.                |
+-----------------------------------+---------------------------------------------+
| | **tx_pulse_len**                | | Length of the transmit pulse in           | 
| | *uint32*                        | | microseconds.                             |
+-----------------------------------+---------------------------------------------+

----------------------
antennas_iq site files
----------------------

Site files are produced by the Borealis code package and have the data in a record by record style format. In site files, the hdf5 group names (ie record names) are given as the timestamp in ms past epoch of the first sequence or sampling period recorded in the record. 

The naming convention of the antennas_iq site-structured files are:

[YYYYmmDD].[HHMM].[SS].[station_id].[slice_id].antennas_iq.hdf5.site

For example: 20191105.1400.02.sas.0.antennas_iq.hdf5.site
This is the file that began writing at 14:00:02 UT on November 5 2019 at the Saskatoon site, and it provides data for slice 0 of the experiment that ran at that time. 

These files are often bzipped after they are produced.

The file fields under the record name in antennas_iq site files are:

+----------------------------------+---------------------------------------------+
| | **Field name**                 | **description**                             |
| | *type*                         |                                             |  
+==================================+=============================================+
| | **antenna_arrays_order**       | | States what order the data is in and      | 
| | *[unicode, ]*                  | | describes the data layout for the         |
| |                                | | num_antennas data dimension. Antennas are |
| |                                | | recorded main array ascending and then    | 
| |                                | | interferometer array ascending.           |
+----------------------------------+---------------------------------------------+
| | **beam_azms**                  | | A list of the beam azimuths for each      |
| | *[float64, ]*                  | | beam in degrees off boresite.             |
+----------------------------------+---------------------------------------------+
| | **beam_nums**                  | | A list of beam numbers used in this slice | 
| | *[uint32, ]*                   | | in this record.                           |
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
| | *[unicode, ]*                  | | represents. = ‘num_antennas’,             |
| |                                | | ‘num_sequences’, ‘num_samps’ for          |
| |                                | | antennas_iq                               |
+----------------------------------+---------------------------------------------+
| | **data_dimensions**            | | The dimensions in which to reshape the    | 
| | *[uint32, ]*                   | | data. Dimensions correspond to            |
| |                                | | data_descriptors.                         |
+----------------------------------+---------------------------------------------+
| | **data_normalization_factor**  | | Scale of all the filters used, multiplied |
| | *float32*                      | | for a total scale to normalize the data   |
| |                                | | by.                                       |
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
| | **freq**                       | | The frequency used for this experiment,   | 
| | *uint32*                       | | in kHz. This is the frequency the data    | 
| |                                | | has been filtered to.                     |
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
| | **noise_at_freq**              | | Noise at the receive frequency, with      | 
| | *[float64, ]*                  | | dimension = number of sequences.          | 
| |                                | | 20191114: not currently implemented and   | 
| |                                | | filled with zeros. Still a TODO.          |
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
| | **pulse_phase_offset**         | | For pulse encoding phase, in degrees      | 
| | *[float32, ]*                  | | offset. Contains one phase offset per     | 
| |                                | | pulse in pulses.                          |
+----------------------------------+---------------------------------------------+
| | **pulses**                     | | The pulse sequence in units of the        | 
| | *[uint32, ]*                   | | tau_spacing.                              |
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
| | **slice_comment**              | | Additional text comment that describes    |
| | *unicode*                      | | the slice written in this file.           |
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
| | **tau_spacing**                | | The minimum spacing between pulses in     | 
| | *uint32*                       | | microseconds. Spacing between pulses is   | 
| |                                | | always a multiple of this.                |
+----------------------------------+---------------------------------------------+
| | **tx_pulse_len**               | | Length of the transmit pulse in           | 
| | *uint32*                       | | microseconds.                             |
+----------------------------------+---------------------------------------------+

------------------------
Site/Array Restructuring
------------------------

File restructuring to array files is done using an additional code package. Currently, this code is housed within `pyDARNio <https://github.com/SuperDARN/pyDARNio>`_.

The site to array file restructuring occurs in the borealis BaseFormat _site_to_array class method, and array to site restructuring is done in the same class _array_to_site method. Both can be found `here <https://github.com/SuperDARN/pyDARNio/blob/master/pydarnio/borealis/borealis_formats.py>`_.
