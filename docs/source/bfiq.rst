=========
bfiq v0.6
=========

This is the most up to date version of this file format produced by Borealis version 0.6, the current version.

For data files from previous Borealis software versions, see `here <https://borealis.readthedocs.io/en/latest/borealis_data.html#previous-versions>`_.

The pyDARNio format class for this format is BorealisBfiq found in the `borealis_formats <https://github.com/SuperDARN/pyDARNio/blob/master/pydarnio/borealis/borealis_formats.py>`_.

The bfiq format is intended to hold beamformed I and Q data for the main and interferometer arrays. The data is not averaged. 

Both site files and array-restructured files exist for this file type. Both are described below.

----------------
bfiq array files
----------------

Array restructured files are produced after the radar has finished writing a file and contain record data in multi-dimensional arrays so as to avoid repeated values, shorten the read time, and improve human readability. Fields that are unique to the record are written as arrays where the first dimension is equal to the number of records recorded. Other fields that are unique to the slice or experiment (and are therefore repeated for all records) are written only once. 

The group names in these files are the field names themselves, greatly reducing the number of group names in the file when compared to site files and making the file much more human readable.

The naming convention of the bfiq array-structured files are:

[YYYYmmDD].[HHMM].[SS].[station_id].[slice_id].bfiq.hdf5

For example: 20191105.1400.02.sas.0.bfiq.hdf5

This is the file that began writing at 14:00:02 UT on November 5 2019 at the Saskatoon site, and it provides data for slice 0 of the experiment that ran at that time. It has been array restructured because it does not have a .site designation at the end of the filename.

These files are zlib compressed which is native to hdf5 and no decompression is necessary before reading using your hdf5 library. 

The file fields in the bfiq array files are:

+-----------------------------------+---------------------------------------------+
| | **FIELD NAME**                  | **description**                             |
| | *type*                          |                                             |
| | [dimensions]                    |                                             |
+===================================+=============================================+
| | **antenna_arrays_order**        | | States what order the data is in and      |
| | *unicode*                       | | describes the data layout for the         |
| | [num_antenna_arrays]            | | num_antenna_arrays data dimension         |
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
| | **blanked_samples**             | | Samples that should be blanked because    |
| | *uint32*                        | | they occurred during transmission times,  |
| | [num_records x                  | | given by sample number (index into        |
| | max_num_blanked_samples ]       | | decimated data). Can differ from the      |
| |                                 | | pulses array due to multiple slices in a  |
| |                                 | | single sequence and can differ from       |
| |                                 | | record to record if a new slice is added. |
+-----------------------------------+---------------------------------------------+
| | **borealis_git_hash**           | | Identifies the version of Borealis that   |
| | *unicode*                       | | made this data. Contains git commit hash  |
| |                                 | | characters. Typically begins with the     |
| |                                 | | latest git tag of the software.           |
+-----------------------------------+---------------------------------------------+
| | **data**                        | | A set of samples (complex float) at given |
| | *complex64*                     | | sample rate. Note that records that do not|
| | [num_records x                  | | have num_sequences = max_num_sequences or |
| | num_antenna_arrays x            | | num_beams = max_num_beams will have       |
| | max_num_sequences x             | | padded zeros. The num_sequences and       |
| | max_num_beams x                 | | num_beams arrays should be used to        |
| | num_samps]                      | | determine the correct number of sequences |
| |                                 | | and beams to read for the record.         |
+-----------------------------------+---------------------------------------------+
| | **data_descriptors**            | | Denotes what each data dimension          |
| | *unicode*                       | | represents. = 'num_records',              |
| | [5]                             | | ‘num_antenna_arrays’,                     |
| |                                 | | ‘max_num_sequences’, ‘max_num_beams’,     |
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
| | **first_range**                 | | Distance to use for first range in km.    |
| | *float32*                       | |                                           | 
+-----------------------------------+---------------------------------------------+
| | **first_range_rtt**             | | Round trip time of flight to first range  |
| | *float32*                       | | in microseconds.                          |
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
| | **lags**                        | | The lags created from two pulses in the   |
| | *uint32*                        | | pulses array. Values have to be from      |
| | [number of lags, 2]             | | pulses array. The lag number is lag[1] -  |
| |                                 | | lag[0] for each lag pair.                 |
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
| | **num_beams**                   | | The number of beams calculated for each   |
| | *uint32*                        | | record. Allows the user to correctly read |
| | [num_records]                   | | the data up to the correct number and     |
| |                                 | | remove the padded zeros in the data       |
| |                                 | | array.                                    |
+-----------------------------------+---------------------------------------------+
| | **num_blanked_samples**         | | The number of blanked samples for each    |
| | *uint32*                        | | record.                                   | 
| | [num_records]                   | |                                           |  
+-----------------------------------+---------------------------------------------+
| | **num_ranges**                  | | Number of ranges to calculate             |
| | *uint32*                        | | correlations for.                         |
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
| | **range_sep**                   | | Range gate separation (conversion from    |
| | *float32*                       | | time (1/rx_sample_rate) to equivalent     |
| |                                 | | distance between samples), in km.         |
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
| | **scheduling_mode**             | | The mode being run during this time       | 
| | *unicode*                       | | period (ex. 'common', 'special',          |
| |                                 | | 'discretionary').                         |
+-----------------------------------+---------------------------------------------+
| | **slice_comment**               | | Additional text comment that describes    |
| | *unicode*                       | | the slice written in this file. The slice |
| |                                 | | number of this file is provided in the    |
| |                                 | | filename.                                 | 
+-----------------------------------+---------------------------------------------+
| | **slice_id**                    | | The slice id of this file.                |
| | *uint32*                        | |                                           |
+-----------------------------------+---------------------------------------------+ 
| | **slice_interfacing**           | | The interfacing of this slice to          | 
| | *unicode*                       | | other slices for each record. String      |
| | [num_records]                   | | representation of the python dictionary   | 
| |                                 | | of {slice : interface_type, ... }. Can    | 
| |                                 | | differ between records if slices updated. | 
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

---------------
bfiq site files
---------------

Site files are produced by the Borealis code package and have the data in a record by record style format. In site files, the hdf5 group names (ie record names) are given as the timestamp in ms past epoch of the first sequence or sampling period recorded in the record. 

The naming convention of the bfiq site-structured files are:

[YYYYmmDD].[HHMM].[SS].[station_id].[slice_id].bfiq.hdf5.site

For example: 20191105.1400.02.sas.0.bfiq.hdf5.site
This is the file that began writing at 14:00:02 UT on November 5 2019 at the Saskatoon site, and it provides data for slice 0 of the experiment that ran at that time. 

These files are often bzipped after they are produced.

The file fields under the record name in bfiq site files are:

+----------------------------------+---------------------------------------------+
| | **Field name**                 | **description**                             |
| | *type*                         |                                             |  
+==================================+=============================================+
| | **antenna_arrays_order**       | | States what order the data is in and      | 
| | *[unicode, ]*                  | | describes the data layout for the         |
| |                                | | num_antenna_arrays data dimension         |
+----------------------------------+---------------------------------------------+
| | **beam_azms**                  | | A list of the beam azimuths for each      |
| | *[float64, ]*                  | | beam in degrees off boresite.             |
+----------------------------------+---------------------------------------------+
| | **beam_nums**                  | | A list of beam numbers used in this slice | 
| | *[uint32, ]*                   | | in this record.                           |
+----------------------------------+---------------------------------------------+
| | **blanked_samples**            | | Samples that should be blanked because    |
| | *[uint32, ]*                   | | they occurred during transmission times,  |
| |                                | | given by sample number (index into        |
| |                                | | decimated data). Can differ from the      |
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
| | *[unicode, ]*                  | | represents. = ‘num_antenna_arrays’,       | 
| |                                | | ‘num_sequences’, ‘num_beams’, ‘num_samps’ | 
| |                                | | for bfiq                                  |
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
| | **first_range**                | | Distance to use for first range in km.    |
| | *float32*                      | |                                           | 
+----------------------------------+---------------------------------------------+
| | **first_range_rtt**            | | Round trip time of flight to first range  | 
| | *float32*                      | | in microseconds.                          |
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
| | **lags**                       | | The lags created from two pulses in the   | 
| | *[[uint32, ], ]*               | | pulses array. Dimensions are number of    | 
| |                                | | lags x 2. Values have to be from pulses   | 
| |                                | | array. The lag number is lag[1] - lag[0]  | 
| |                                | | for each lag pair.                        |
+----------------------------------+---------------------------------------------+
| | **main_antenna_count**         | | Number of main array antennas             |
| | *uint32*                       | |                                           | 
+----------------------------------+---------------------------------------------+
| | **noise_at_freq**              | | Noise at the receive frequency, with      | 
| | *[float64, ]*                  | | dimension = number of sequences.          | 
| |                                | | 20191114: not currently implemented and   | 
| |                                | | filled with zeros. Still a TODO.          |
+----------------------------------+---------------------------------------------+
| | **num_ranges**                 | | Number of ranges to calculate             | 
| | *uint32*                       | | correlations for.                         |
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
| | **range_sep**                  | | Range gate separation (conversion from    | 
| | *float32*                      | | time (1/rx_sample_rate) to equivalent     | 
| |                                | | distance between samples), in km.         |
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
| | **slice_comment**              | | Additional text comment that describes    |
| | *unicode*                      | | the slice written in this file.           |
+----------------------------------+---------------------------------------------+
| | **slice_id**                   | | The slice id of this file.                |
| | *uint32*                       | |                                           |
+----------------------------------+---------------------------------------------+ 
| | **slice_interfacing**          | | The interfacing of this slice to          | 
| | *unicode*                      | | other slices. String representation of    |
| |                                | | the python dictionary of                  | 
| |                                | | {slice : interface_type, ... }            | 
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

-------------------------------------
bfiq to iqdat SDARN (DMap) Conversion
-------------------------------------

Conversion to SDARN IO (DMap iqdat) is available but can fail based on experiment complexity. The conversion also reduces the precision of the data due to conversion from complex floats to int of all samples. Similar precision is lost in timestamps. 

HDF5 is a much more user-friendly format and we encourage the use of this data if possible. Please reach out if you have questions on how to use the Borealis bfiq files.

The mapping from bfiq to iqdat dmap files is completed as follows:

..  toctree::
    :maxdepth: 2

    iqdat_mapping
