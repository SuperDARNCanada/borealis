===========
antennas_iq
===========

This format is intended to hold individual antennas i and q data. The data is not averaged. 

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

| **Field name** | **numpy type** | **array dimensions** | **description** |
| --- | --- | --- | --- |
| antenna_arrays_order | unicode | num_antennas | States what order the data is in and describes the data layout for the num_antennas data dimension. Antennas are recorded main array ascending and then interferometer array ascending. |
| beam_azms | float64 | num_records x max_num_beams (in any record) | A list of the beam azimuths for each beam in degrees off boresite for each record. Note that this is padded with zeroes for any record which has num_beams less than the max_num_beams. The num_beams field should be used to read the correct number of beams for each record. |
| beam_nums | uint32 | num_records x max_num_beams (in any record) | A list of beam numbers used in this slice in this record. Note that this is padded with zeroes for any record which has num_beams less than the max_num_beams. The num_beams field should be used to read the correct number of beams for each record. |
| borealis_git_hash | unicode | non-array | Identifies the version of Borealis that made this data. Contains git commit hash characters. Typically begins with the latest git tag of the software. |
| data | complex64 | num_records x num_antennas x max_num_sequences x num_samps | A set of samples (complex float) at given sample rate. Note that records that do not have num_sequences = max_num_sequences will have padded zeros. The num_sequences array should be used to determine the correct number of sequences to read for the record. |
| data_descriptors | unicode | 4 | Denotes what each data dimension represents. = 'num_records', ‘num_antennas’, ‘max_num_sequences’, ‘num_samps’ |
| data_normalization_factor | float32 | non-array | Scale of all the filters used, multiplied, for a total scale to normalize the data by. |
| experiment_comment | unicode | non-array | Comment provided in experiment about the experiment as a whole. |
| experiment_id | int64 | non-array| Number used to identify the experiment. |
| experiment_name | unicode | non-array | Name of the experiment file. |
| freq | uint32 | non-array | The frequency used for this experiment, in kHz. This is the frequency the data has been filtered to. |
| int_time | float32 | num_records | Integration time in seconds. |
| intf_antenna_count | uint32 | non-array | Number of interferometer array antennas |
| main_antenna_count | uint32 | non-array | Number of main array antennas |
| noise_at_freq | float64 | num_records x max_num_sequences | Noise at the receive frequency, with dimension = number of sequences. 20191114: not currently implemented and filled with zeros. Still a TODO. Note that records that do not have num_sequences = max_num_sequences will have padded zeros. The num_sequences array should be used to determine the correct number of sequences to read for the record. |
| num_beams | uint32 | num_records | The number of beams calculated for each record. Allows the user to correctly read the data up to the correct number and remove the padded zeros in the data array. | 
| num_samps | uint32 | non-array | Number of samples in the sampling periods. Will also be provided as the last data_dimension value. |
| num_sequences | int64 | num_records | Number of sampling periods (equivalent to number sequences transmitted) in the integration time for each record. Allows the user to correctly read the data up to the correct number and remove the padded zeros in the data array. |
| num_slices | int64 | num_records | Number of slices used simultaneously in the record by the experiment. If more than 1, data should exist in another file for the same time period as that record for the other slice. |
| pulse_phase_offset | float32 | number of pulses | For pulse encoding phase, in degrees offset. Contains one phase offset per pulse in pulses. |
| pulses | uint32 | number of pulses | The pulse sequence in units of the tau_spacing. |
| rx_sample_rate | float64 | non-array | Sampling rate of the samples in this file's data in Hz. |
| samples_data_type | unicode | non-array | C data type of the samples, provided for user friendliness. = 'complex float' |
| scan_start_marker | bool | num_records | Designates if the record is the first in a scan (scan is defined by the experiment). |
| slice_comment | unicode | non-array | Additional text comment that describes the slice written in this file. The slice number of this file is provided in the filename. |
| sqn_timestamps | float64 | num_records x max_num_sequences | A list of GPS timestamps corresponding to the beginning of transmission for each sampling period in the integration time. These timestamps come from the USRP driver and the USRPs are GPS disciplined and synchronized using the Octoclock. Provided in milliseconds since epoch. Note that records that do not have num_sequences = max_num_sequences will have padded zeros. The num_sequences array should be used to determine the correct number of sequences to read for the record. |
| station | unicode | non-array | Three-letter radar identifier. |
| tau_spacing | uint32 | non-array | The minimum spacing between pulses in microseconds. Spacing between pulses is always a multiple of this. |
| tx_pulse_len | uint32 | non-array | Length of the transmit pulse in microseconds. |


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

| **Field name** | **numpy type** | **description** |
| --- | --- | --- |
| antenna_arrays_order | [unicode, ] | States what order the data is in and describes the data layout for the num_antennas data dimension. Antennas are recorded main array ascending and then interferometer array ascending. |
| beam_azms | [float64, ] | A list of the beam azimuths for each beam in degrees off boresite. |
| beam_nums | [uint32, ] | A list of beam numbers used in this slice in this record. |
| borealis_git_hash | unicode | Identifies the version of Borealis that made this data. Contains git commit hash characters. Typically begins with the latest git tag of the software. |
| data | [complex64, ] | A contiguous set of samples (complex float) at given sample rate. Needs to be reshaped by data_dimensions to be correctly read. |
| data_descriptors | [unicode, ] | Denotes what each data dimension represents. = ‘num_antennas’, ‘num_sequences’, ‘num_samps’ for bfiq |
| data_dimensions | [uint32, ] | The dimensions in which to reshape the data. Dimensions correspond to data_descriptors. |
| data_normalization_factor | float32 | Scale of all the filters used, multiplied, for a total scale to normalize the data by. |
| experiment_comment | unicode | Comment provided in experiment about the experiment as a whole. |
| experiment_id | int64 | Number used to identify the experiment. |
| experiment_name | unicode | Name of the experiment file. |
| freq | uint32 | The frequency used for this experiment, in kHz. This is the frequency the data has been filtered to. |
| int_time | float32 | Integration time in seconds. |
| intf_antenna_count | uint32 | Number of interferometer array antennas |
| main_antenna_count | uint32 | Number of main array antennas |
| noise_at_freq | [float64, ] | Noise at the receive frequency, with dimension = number of sequences. 20191114: not currently implemented and filled with zeros. Still a TODO. |
| num_samps | uint32 | Number of samples in the sampling period. Will also be provided as the last data_dimension value. |
| num_sequences | int64 | Number of sampling periods (equivalent to number sequences transmitted) in the integration time. |
| num_slices | int64 | Number of slices used simultaneously in this record by the experiment. If more than 1, data should exist in another file for this time period for the other slice. |
| pulse_phase_offset | [float32, ] | For pulse encoding phase, in degrees offset. Contains one phase offset per pulse in pulses. |
| pulses | [uint32, ] | The pulse sequence in units of the tau_spacing. |
| rx_sample_rate | float64 | Sampling rate of the samples in this file's data in Hz. |
| samples_data_type | unicode | C data type of the samples, provided for user friendliness. = 'complex float' |
| scan_start_marker | bool | Designates if the record is the first in a scan (scan is defined by the experiment). |
| slice_comment | unicode | Additional text comment that describes the slice written in this file. The slice number of this file is provided in the filename. |
| sqn_timestamps | [float64, ] | A list of GPS timestamps corresponding to the beginning of transmission for each sampling period in the integration time. These timestamps come from the USRP driver and the USRPs are GPS disciplined and synchronized using the Octoclock. Provided in milliseconds since epoch. |
| station | unicode | Three-letter radar identifier. |
| tau_spacing | uint32 | The minimum spacing between pulses in microseconds. Spacing between pulses is always a multiple of this. |
| tx_pulse_len | uint32 | Length of the transmit pulse in microseconds. |

------------------
File Restructuring
------------------

File restructuring to array files is done using an additional code package. Currently, this code is housed within `pyDARN <https://github.com/SuperDARN/pydarn/tree/feature/borealis_conversion>`_. It is expected that this code will be separated to its own code package in the near future.

The site to array file restructuring occurs here: `Link to Source <https://github.com/SuperDARN/pydarn/blob/feature/borealis_conversion/pydarn/io/borealis/restructure_borealis.py#L295>`_

Array to site restructuring can also be done and is contained within the same file.

