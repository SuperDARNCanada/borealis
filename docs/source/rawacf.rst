======
rawacf
======

This format is intended to hold beamformed, averaged, correlated data. 

Both site files and array-restructured files exist for this file type. Both are described below.

------------------
rawacf array files
------------------

Array restructured files are produced after the radar has finished writing a file and contain record data in multi-dimensional arrays so as to avoid repeated values, shorten the read time, and improve human readability. Fields that are unique to the record are written as arrays where the first dimension is equal to the number of records recorded. Other fields that are unique to the slice or experiment (and are therefore repeated for all records) are written only once. 

The group names in these files are the field names themselves, greatly reducing the number of group names in the file when compared to site files and making the file much more human readable.

The naming convention of the rawacf array-structured files are:

[YYYYmmDD].[HHMM].[SS].[station_id].[slice_id].rawacf.hdf5

For example: 20191105.1400.02.sas.0.rawacf.hdf5
This is the file that began writing at 14:00:02 UT on November 5 2019 at the Saskatoon site, and it provides data for slice 0 of the experiment that ran at that time. It has been array restructured because it does not have a .site designation at the end of the filename.

These files are zlib compressed which is native to hdf5 and no decompression is necessary before reading using your hdf5 library. 

The file fields in the rawacf array files are:

+----------------------+----------------+----------------------+-------------------------------------------+
|    **Field name**    | **numpy type** | **array dimensions** | **description**                           |
+======================+================+======================+===========================================+
| beam_azms            | float64        | num_records x        | A list of the beam azimuths for each beam |
|                      |                | max_num_beams (in    | in degrees off boresite. Note that this   |
|                      |                | any record)          | is padded with zeroes for any record      |
|                      |                |                      | which has num_beams less than the         |
|                      |                |                      | max_num_beams. The num_beams field should | 
|                      |                |                      | be used to read the correct number of     | 
|                      |                |                      | beams for each record.                    |
+----------------------+----------------+----------------------+-------------------------------------------+
| beam_nums            | uint32         | num_records x        | A list of beam numbers used in this slice |
|                      |                | max_num_beams (in    | in this record. Note that this is padded  |
|                      |                | any record)          | with zeroes for any record which has      |
|                      |                |                      | num_beams less than the max_num_beams.    |
|                      |                |                      | The num_beams field should be used to     |
|                      |                |                      | read the correct number of beams for each |
|                      |                |                      | record.                                   |
+----------------------+----------------+----------------------+-------------------------------------------+
| blanked_samples      | uint32         | number of blanked    | Samples that should be blanked because    |
|                      |                | samples              | they occurred during transmission times,  |
|                      |                |                      | given by sample number (index into        |
|                      |                |                      | decimated data). Can differ from the      |
|                      |                |                      | pulses array due to multiple slices in a  |
|                      |                |                      | single sequence.                          |
+----------------------+----------------+----------------------+-------------------------------------------+
| borealis_git_hash    | unicode        | non-array            | Identifies the version of Borealis that   |
|                      |                |                      | made this data. Contains git commit hash  |
|                      |                |                      | characters. Typically begins with the     |
|                      |                |                      | latest git tag of the software.           |
+----------------------+----------------+----------------------+-------------------------------------------+
| correlation_         | unicode        | 5                    | Denotes what each correlation dimension   |
| descriptors          |                |                      | (in main_acfs, intf_acfs, xcfs)           |
|                      |                |                      | represents. = 'num_records'               | 
|                      |                |                      | 'max_num_beams' 'num_ranges' 'num_lags'   | 
+----------------------+----------------+----------------------+-------------------------------------------+
| data_normalization_  | float32        | non-array            | Scale of all the filters used,            |
| factor               |                |                      | multiplied, for a total scale to          | 
|                      |                |                      | normalize the data by.                    |
+----------------------+----------------+----------------------+-------------------------------------------+
| experiment_comment   | unicode        | non-array            | Comment provided in experiment about the  | 
|                      |                |                      | experiment as a whole.                    |
+----------------------+----------------+----------------------+-------------------------------------------+
| experiment_id        | int64          | non-array            | Number used to identify the experiment.   |
+----------------------+----------------+----------------------+-------------------------------------------+
| experiment_name      | unicode        | non-array            | Name of the experiment file.              |
+----------------------+----------------+----------------------+-------------------------------------------+
| first_range          | float32        | non-array            | Distance to use for first range in km.    |
+----------------------+----------------+----------------------+-------------------------------------------+
| first_range_rtt      | float32        | non-array            | Round trip time of flight to first range  |
|                      |                |                      | in microseconds.                          |
+----------------------+----------------+----------------------+-------------------------------------------+
| freq                 | uint32         | non-array            | The frequency used for this experiment,   |
|                      |                |                      | in kHz. This is the frequency the data    |
|                      |                |                      | has been filtered to.                     |
+----------------------+----------------+----------------------+-------------------------------------------+
| int_time             | float32        | num_records          | Integration time in seconds.              |
+----------------------+----------------+----------------------+-------------------------------------------+
| intf_acfs            | complex64      | num_records x        | Interferometer array correlations. Note   |
|                      |                | max_num_beams x      | that records that do not have num_beams = |
|                      |                | num_ranges x         | max_num_beams will have padded zeros. The |
|                      |                | num_lags             | num_beams array should be used to         | 
|                      |                |                      | determine the correct number of beams to  | 
|                      |                |                      | read for the record.                      |
+----------------------+----------------+----------------------+-------------------------------------------+
| intf_antenna_count   | uint32         | non-array            | Number of interferometer array antennas   |
+----------------------+----------------+----------------------+-------------------------------------------+
| lags                 | uint32         | number of lags x 2   | The lags created from two pulses in the   |
|                      |                |                      | pulses array. Values have to be from      |
|                      |                |                      | pulses array. The lag number is lag[1] -  |
|                      |                |                      | lag[0] for each lag pair.                 |
+----------------------+----------------+----------------------+-------------------------------------------+
| main_acfs            | complex64      | num_records x        | Main array correlations. Note             |
|                      |                | max_num_beams x      | that records that do not have num_beams = |
|                      |                | num_ranges x         | max_num_beams will have padded zeros. The |
|                      |                | num_lags             | num_beams array should be used to         | 
|                      |                |                      | determine the correct number of beams to  | 
|                      |                |                      | read for the record.                      |
+----------------------+----------------+----------------------+-------------------------------------------+
| main_antenna_count   | uint32         | non-array            | Number of main array antennas             |
+----------------------+----------------+----------------------+-------------------------------------------+
| noise_at_freq        | float64        | num_records x        | Noise at the receive frequency, with      |
|                      |                | max_num_sequences    | dimension = number of sequences.          |
|                      |                |                      | 20191114: not currently implemented and   |
|                      |                |                      | filled with zeros. Still a TODO. Note     |
|                      |                |                      | that records that do not have             |
|                      |                |                      | num_sequences = max_num_sequences will    |
|                      |                |                      | have padded zeros. The num_sequences      |
|                      |                |                      | array should be used to determine the     |
|                      |                |                      | correct number of sequences to read for   |
|                      |                |                      | the record.                               |
+----------------------+----------------+----------------------+-------------------------------------------+
| num_beams            | uint32         | num_records          | The number of beams calculated for each   |
|                      |                |                      | record. Allows the user to correctly read |
|                      |                |                      | the data up to the correct number and     |
|                      |                |                      | remove the padded zeros in the data       |
|                      |                |                      | array.                                    | 
+----------------------+----------------+----------------------+-------------------------------------------+
| num_sequences        | int64          | num_records          | Number of sampling periods (equivalent to |
|                      |                |                      | number sequences transmitted) in the      | 
|                      |                |                      | integration time for each record. Allows  | 
|                      |                |                      | the user to correctly read the data up to |
|                      |                |                      | the correct number and remove the padded  |
|                      |                |                      | zeros in the data array.                  |
+----------------------+----------------+----------------------+-------------------------------------------+
| num_slices           | int64          | num_records          | Number of slices used simultaneously in   |
|                      |                |                      | the record by the experiment. If more     |
|                      |                |                      | than 1, data should exist in another file |
|                      |                |                      | for the same time period as that record   |
|                      |                |                      | for the other slice.                      |
+----------------------+----------------+----------------------+-------------------------------------------+
| pulses               | uint32         | number of pulses     | The pulse sequence in units of the        |
|                      |                |                      | tau_spacing.                              |
+----------------------+----------------+----------------------+-------------------------------------------+
| range_sep            | float32        | non-array            | Range gate separation (conversion from    |
|                      |                |                      | time (1/rx_sample_rate) to equivalent     |
|                      |                |                      | distance between samples), in km.         |
+----------------------+----------------+----------------------+-------------------------------------------+
| rx_sample_rate       | float64        | non-array            | Sampling rate of the samples in this      |
|                      |                |                      | file's data in Hz.                        |
+----------------------+----------------+----------------------+-------------------------------------------+
| samples_data_type    | unicode        | non-array            | C data type of the samples, provided for  |
|                      |                |                      | user friendliness. = 'complex float'      |
+----------------------+----------------+----------------------+-------------------------------------------+
| scan_start_marker    | bool           | num_records          | Designates if the record is the first in  | 
|                      |                |                      | a scan (scan is defined by the            |
|                      |                |                      | experiment).                              |
+----------------------+----------------+----------------------+-------------------------------------------+
| slice_comment        | unicode        | non-array            | Additional text comment that describes    |
|                      |                |                      | the slice written in this file. The slice |
|                      |                |                      | number of this file is provided in the    |
|                      |                |                      | filename.                                 | 
+----------------------+----------------+----------------------+-------------------------------------------+
| sqn_timestamps       | float64        | num_records x        | A list of GPS timestamps corresponding to |
|                      |                | max_num_sequences    | the beginning of transmission for each    | 
|                      |                |                      | sampling period in the integration time.  |
|                      |                |                      | These timestamps come back from the USRP  | 
|                      |                |                      | driver and the USRPs are GPS disciplined  |
|                      |                |                      | and synchronized using the Octoclock.     |
|                      |                |                      | Provided in milliseconds since epoch.     | 
|                      |                |                      | Note that records that do not have        | 
|                      |                |                      | num_sequences = max_num_sequences will    | 
|                      |                |                      | have padded zeros. The num_sequences      | 
|                      |                |                      | array should be used to determine the     | 
|                      |                |                      | correct number of sequences to read for   | 
|                      |                |                      | the record.                               |
+----------------------+----------------+----------------------+-------------------------------------------+
| station              | unicode        | non-array            | Three-letter radar identifier.            |
+----------------------+----------------+----------------------+-------------------------------------------+
| tau_spacing          | uint32         | non-array            | The minimum spacing between pulses in     | 
|                      |                |                      | microseconds. Spacing between pulses is   | 
|                      |                |                      | always a multiple of this.                |
+----------------------+----------------+----------------------+-------------------------------------------+
| tx_pulse_len         | uint32         | non-array            | Length of the transmit pulse in           | 
|                      |                |                      | microseconds.                             |
+----------------------+----------------+----------------------+-------------------------------------------+
| xcfs                 | complex64      | num_records x        | Cross correlations of interferometer to   | 
|                      |                | max_num_beams x      | main array. Note                          |
|                      |                | num_ranges x         | that records that do not have num_beams = |
|                      |                | num_lags             | max_num_beams will have padded zeros. The |
|                      |                |                      | num_beams array should be used to         | 
|                      |                |                      | determine the correct number of beams to  | 
|                      |                |                      | read for the record.                      |
+----------------------+----------------+----------------------+-------------------------------------------+

-----------------
rawacf site files
-----------------

Site files are produced by the Borealis code package and have the data in a record by record style format. In site files, the hdf5 group names (ie record names) are given as the timestamp in ms past epoch of the first sequence or sampling period recorded in the record. 

The naming convention of the rawacf site-structured files are:

[YYYYmmDD].[HHMM].[SS].[station_id].[slice_id].rawacf.hdf5.site

For example: 20191105.1400.02.sas.0.rawacf.hdf5.site
This is the file that began writing at 14:00:02 UT on November 5 2019 at the Saskatoon site, and it provides data for slice 0 of the experiment that ran at that time. 

These files are often bzipped after they are produced.

The file fields under the record name in rawacf site files are:

+----------------------+----------------+-------------------------------------------+
|    **Field name**    | **numpy type** | **description**                           |
+======================+================+===========================================+
| beam_azms            | [float64, ]    | A list of the beam azimuths for each      |
|                      |                | beam in degrees off boresite.             |
+----------------------+----------------+-------------------------------------------+
| beam_nums            | [uint32, ]     | A list of beam numbers used in this slice | 
|                      |                | in this record.                           |
+----------------------+----------------+-------------------------------------------+
| blanked_samples      | [uint32, ]     | Samples that should be blanked because    | 
|                      |                | they occurred during transmission times,  | 
|                      |                | given by sample number (index into        | 
|                      |                | decimated data). Can differ from the      | 
|                      |                | pulses array due to multiple slices in a  | 
|                      |                | single sequence.                          |
+----------------------+----------------+-------------------------------------------+
| borealis_git_hash    | unicode        | Identifies the version of Borealis that   | 
|                      |                | made this data. Contains git commit hash  | 
|                      |                | characters. Typically begins with the     | 
|                      |                | latest git tag of the software.           |
+----------------------+----------------+-------------------------------------------+
| correlation_         | [unicode, ]    | Denotes what each correlation dimension   | 
| descriptors          |                | (in main_acfs, intf_acfs, xcfs)           | 
|                      |                | represents ('num_beams' 'num_ranges'      | 
|                      |                | 'num_lags')                               | 
+----------------------+----------------+-------------------------------------------+
| correlation_         | [uint32, ]     | The dimensions in which to reshape the    | 
| dimensions           |                | acf or xcf datasets.                      |
+----------------------+----------------+-------------------------------------------+
| data_normalization_  | float32        | Scale of all the filters used, multiplied |
| factor               |                | for a total scale to normalize the data   |
|                      |                | by.                                       |
+----------------------+----------------+-------------------------------------------+
| experiment_comment   | unicode        | Comment provided in experiment about the  |
|                      |                | experiment as a whole.                    |
+----------------------+----------------+-------------------------------------------+
| experiment_id        | int64          | Number used to identify the experiment.   |
+----------------------+----------------+-------------------------------------------+
| experiment_name      | unicode        | Name of the experiment file.              |
+----------------------+----------------+-------------------------------------------+
| first_range          | float32        | Distance to use for first range in km.    |
+----------------------+----------------+-------------------------------------------+
| first_range_rtt      | float32        | Round trip time of flight to first range  | 
|                      |                | in microseconds.                          |
+----------------------+----------------+-------------------------------------------+
| freq                 | uint32         | The frequency used for this experiment,   | 
|                      |                | in kHz. This is the frequency the data    | 
|                      |                | has been filtered to.                     |
+----------------------+----------------+-------------------------------------------+
| int_time             | float32        | Integration time in seconds.              |
+----------------------+----------------+-------------------------------------------+
| intf_acfs            | [complex64, ]  | Interferometer array correlations.        |
+----------------------+----------------+-------------------------------------------+
| intf_antenna_count   | uint32         | Number of interferometer array antennas   |
+----------------------+----------------+-------------------------------------------+
| lags                 | [[uint32, ], ] | The lags created from two pulses in the   | 
|                      |                | pulses array. Dimensions are number of    | 
|                      |                | lags x 2. Values have to be from pulses   | 
|                      |                | array. The lag number is lag[1] - lag[0]  | 
|                      |                | for each lag pair.                        |
+----------------------+----------------+-------------------------------------------+
| main_acfs            | [complex64, ]  | Main array correlations.                  |
+----------------------+----------------+-------------------------------------------+
| main_antenna_count   | uint32         | Number of main array antennas             |
+----------------------+----------------+-------------------------------------------+
| noise_at_freq        | [float64, ]    | Noise at the receive frequency, with      | 
|                      |                | dimension = number of sequences.          | 
|                      |                | 20191114: not currently implemented and   | 
|                      |                | filled with zeros. Still a TODO.          |
+----------------------+----------------+-------------------------------------------+
| num_sequences        | int64          | Number of sampling periods (equivalent to | 
|                      |                | number sequences transmitted) in the      | 
|                      |                | integration time.                         |
+----------------------+----------------+-------------------------------------------+
| num_slices           | int64          | Number of slices used simultaneously in   | 
|                      |                | this record by the experiment. If more    | 
|                      |                | than 1, data should exist in another file | 
|                      |                | for this time period for the other slice. |
+----------------------+----------------+-------------------------------------------+
| pulses               | [uint32, ]     | The pulse sequence in units of the        | 
|                      |                | tau_spacing.                              |
+----------------------+----------------+-------------------------------------------+
| range_sep            | float32        | Range gate separation (conversion from    | 
|                      |                | time (1/rx_sample_rate) to equivalent     | 
|                      |                | distance between samples), in km.         |
+----------------------+----------------+-------------------------------------------+
| rx_sample_rate       | float64        | Sampling rate of the samples in this      | 
|                      |                | file's data in Hz.                        |
+----------------------+----------------+-------------------------------------------+
| samples_data_type    | unicode        | C data type of the samples, provided for  | 
|                      |                | user friendliness. = 'complex float'      |
+----------------------+----------------+-------------------------------------------+
| scan_start_marker    | bool           | Designates if the record is the first in  | 
|                      |                | a scan (scan is defined by the            | 
|                      |                | experiment).                              |
+----------------------+----------------+-------------------------------------------+
| slice_comment        | unicode        | Additional text comment that describes    | 
|                      |                | the slice written in this file. The slice | 
|                      |                | number of this file is provided in the    | 
|                      |                | filename.                                 |
+----------------------+----------------+-------------------------------------------+
| sqn_timestamps       | [float64, ]    | A list of GPS timestamps corresponding to | 
|                      |                | the beginning of transmission for each    | 
|                      |                | sampling period in the integration time.  | 
|                      |                | These timestamps come from the USRP       | 
|                      |                | driver and the USRPs are GPS disciplined  | 
|                      |                | and synchronized using the Octoclock.     | 
|                      |                | Provided in milliseconds since epoch.     |
+----------------------+----------------+-------------------------------------------+
| station              | unicode        | Three-letter radar identifier.            |
+----------------------+----------------+-------------------------------------------+
| tau_spacing          | uint32         | The minimum spacing between pulses in     | 
|                      |                | microseconds. Spacing between pulses is   | 
|                      |                | always a multiple of this.                |
+----------------------+----------------+-------------------------------------------+
| tx_pulse_len         | uint32         | Length of the transmit pulse in           | 
|                      |                | microseconds.                             |
+----------------------+----------------+-------------------------------------------+
| xcfs                 | [complex64, ]  | Cross correlations of interferometer to   | 
|                      |                | main array.                               |
+----------------------+----------------+-------------------------------------------+

------------------
File Restructuring
------------------

File restructuring to array files is done using an additional code package. Currently, this code is housed within `pyDARN <https://github.com/SuperDARN/pydarn/tree/feature/borealis_conversion>`_. It is expected that this code will be separated to its own code package in the near future.

The site to array file restructuring occurs here: `Link to Source <https://github.com/SuperDARN/pydarn/blob/feature/borealis_conversion/pydarn/io/borealis/restructure_borealis.py#L332>`_

Array to site restructuring can also be done and is contained within the same file.

----------------------------------------
rawacf to rawacf SDARN (DMap) Conversion
----------------------------------------

Conversion to SDARN IO (DMap rawacf) is available but can fail based on experiment complexity. The conversion also reduces the precision of the data due to conversion from complex floats to int of all samples. Similar precision is lost in timestamps. 

HDF5 is a much more user-friendly format and we encourage the use of this data if possible. Please reach out if you have questions on how to use the Borealis rawacf files.

The mapping to rawacf dmap files is completed as follows:

..  toctree::
    :maxdepth: 2

    rawacf_mapping
