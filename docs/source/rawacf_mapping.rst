==============
rawacf_mapping
==============

RAWACF SDARN FIELDS

This conversion is done in pyDARNio here in the __convert_rawacf_record method: `Link to Source <https://github.com/SuperDARN/pyDARNio/blob/master/pydarnio/borealis/borealis_convert.py>`_

+-----------------------------------+---------------------------------------------+
| | **SDARN DMAP FIELD NAME**       | **Borealis Conversion**                     |
| | *type*                          |                                             |
| | SDARN description               |                                             |
+===================================+=============================================+
| | **radar.revision.major**        | | *borealis_git_hash* major version number  |
| | *char*                          | | or 255 if not a commit with a version tag |  
| | Major version number            | |                                           |
+-----------------------------------+---------------------------------------------+
| | **radar.revision.minor**        | | *borealis_git_hash* minor version number  |
| | *char*                          | | or 255 if not a commit with a version tag | 
| | Minor version number            | |                                           |
+-----------------------------------+---------------------------------------------+
| | **origin.code**                 | | = 100, this can be used as a flag that the|
| | *char*                          | | origin code was Borealis                  |
| | Code indicating origin of data  | |                                           |
+-----------------------------------+---------------------------------------------+
| | **origin.time**                 | | *timestamp_of_write* conversion           |
| | *string*                        | |                                           |
| | ASCII representation of when    | |                                           |
| | the data was generated          | |                                           |
+-----------------------------------+---------------------------------------------+
| | **origin.command**              | | Borealis vXXX + *borealis_git_hash* +     |
| | *string*                        | | *experiment_name*                         |
| | The command line or control     | |                                           |
| | program used to generate the    | |                                           |
| | data                            | |                                           |
+-----------------------------------+---------------------------------------------+
| | **cp**                          | | *experiment_id*, truncated to short       |
| | *short*                         | |                                           | 
| | Control program identifier      | |                                           |
+-----------------------------------+---------------------------------------------+
| | **stid**                        | | *station* conversion                      |
| | *short*                         | |                                           |
| | Station identifier              | |                                           |
+-----------------------------------+---------------------------------------------+
| | **time.yr**                     | | *sqn_timestamps* [0] conversion           |
| | *short*                         | |                                           |      
| | Year                            | |                                           |
+-----------------------------------+---------------------------------------------+
| | **time.mo**                     | | *sqn_timestamps* [0] conversion           |
| | *short*                         | |                                           |
| | Month                           | |                                           |
+-----------------------------------+---------------------------------------------+
| | **time.dy**                     | | *sqn_timestamps* [0] conversion           |
| | *short*                         | |                                           |
| | Day                             | |                                           |
+-----------------------------------+---------------------------------------------+
| | **time.hr**                     | | *sqn_timestamps* [0] conversion           |
| | *short*                         | |                                           |      
| | Hour                            | |                                           |
+-----------------------------------+---------------------------------------------+
| | **time.mt**                     | | *sqn_timestamps* [0] conversion           |
| | *short*                         | |                                           |
| | Minute                          | |                                           |
+-----------------------------------+---------------------------------------------+
| | **time.sc**                     | | *sqn_timestamps* [0] conversion           |
| | *short*                         | |                                           |
| | Second                          | |                                           |
+-----------------------------------+---------------------------------------------+
| | **time.us**                     | | *sqn_timestamps* [0] conversion           |
| | *short*                         | |                                           |
| | Microsecond                     | |                                           |
+-----------------------------------+---------------------------------------------+
| | **txpow**                       | | = -1 (filler)                             |
| | *short*                         | |                                           |
| | Transmitted power (kW)          | |                                           |
+-----------------------------------+---------------------------------------------+
| | **nave**                        | | *num_sequences*                           |
| | *short*                         | |                                           |
| | Number of pulse sequences       | |                                           |
| | transmitted                     | |                                           |
+-----------------------------------+---------------------------------------------+
| | **atten**                       | | = 0 (filler)                              |
| | *short*                         | |                                           |
| | Attenuation level               | |                                           |
+-----------------------------------+---------------------------------------------+
| | **lagfr**                       | | *first_range_rtt*                         |
| | *short*                         | |                                           |
| | Lag to first range              | |                                           |
| | (microseconds)                  | |                                           |
+-----------------------------------+---------------------------------------------+
| | **smsep**                       | | (*rx_sample_rate*)^ -1                    |
| | *short*                         | |                                           |
| | Sample separation               | |                                           |
| | (microseconds)                  | |                                           |
+-----------------------------------+---------------------------------------------+
| | **ercod**                       | | = 0 (filler)                              |
| | *short*                         | |                                           |
| | Error code                      | |                                           |
+-----------------------------------+---------------------------------------------+
| | **stat.agc**                    | | = 0 (filler)                              |
| | *short*                         | |                                           |
| | AGC status word                 | |                                           |
+-----------------------------------+---------------------------------------------+
| | **stat.lopwr**                  | | = 0 (filler)                              |
| | *short*                         | |                                           |
| | LOPWR status word               | |                                           |
+-----------------------------------+---------------------------------------------+
| | **noise.search**                | | *noise_at_freq* [0] conversion            |
| | *float*                         | |                                           |
| | Calculated noise from clear     | |                                           |
| | frequency search                | |                                           |
+-----------------------------------+---------------------------------------------+
| | **noise.mean**                  | | = 0 (filler)                              |
| | *float*                         | |                                           |
| | Average noise across frequency  | |                                           |
| | band                            | |                                           |
+-----------------------------------+---------------------------------------------+
| | **channel**                     | | *slice_id*                                |
| | *short*                         | |                                           |
| | Channel number for a stereo     | |                                           |
| | radar (zero for all others)     | |                                           |
+-----------------------------------+---------------------------------------------+
| | **bmnum**                       | | *beam_nums* [i]                           |
| | *short*                         | |                                           |
| | Beam number                     | |                                           |
+-----------------------------------+---------------------------------------------+
| | **bmazm**                       | | *beam_azms* [i]                           |
| | *float*                         | |                                           |
| | Beam azimuth                    | |                                           |
+-----------------------------------+---------------------------------------------+
| | **scan**                        | | *scan_start_marker* (0 or 1)              |
| | *short*                         | |                                           |
| | Scan flag                       | |                                           |
+-----------------------------------+---------------------------------------------+
| | **offset**                      | | = 0 (filler)                              |
| | *short*                         | |                                           |
| | Offset between channels for a   | |                                           |
| | stereo radar (zero for all      | |                                           |
| | others)                         | |                                           |
+-----------------------------------+---------------------------------------------+
| | **rxrise**                      | | = 0.0                                     |
| | *short*                         | |                                           |
| | Receiver rise time              | |                                           |
| | (microseconds)                  | |                                           |
+-----------------------------------+---------------------------------------------+
| | **intt.sc**                     | | *int_time* conversion                     |
| | *short*                         | |                                           |
| | Whole number of seconds of      | |                                           |
| | integration time.               | |                                           |
+-----------------------------------+---------------------------------------------+
| | **intt.us**                     | | *int_time* conversion                     |
| | *short*                         | |                                           |
| | Fractional number of            | |                                           |
| | microseconds of integration     | |                                           |
| | time                            | |                                           |
+-----------------------------------+---------------------------------------------+
| | **txpl**                        | | *tx_pulse_len*                            |
| | *short*                         | |                                           |
| | Transmit pulse length           | |                                           |
| | (microseconds)                  | |                                           |
+-----------------------------------+---------------------------------------------+
| | **mpinc**                       | | *tau_spacing*                             |
| | *short*                         | |                                           |
| | Multi-pulse increment           | |                                           |
| | (microseconds)                  | |                                           |
+-----------------------------------+---------------------------------------------+
| | **mppul**                       | | len(*pulses*)                             |
| | *short*                         | |                                           |
| | Number of pulses in sequence    | |                                           |
+-----------------------------------+---------------------------------------------+
| | **mplgs**                       | | *lags*.shape[0]                           |
| | *short*                         | |                                           |
| | Number of lags in sequence      | |                                           |
+-----------------------------------+---------------------------------------------+
| | **nrang**                       | | *correlation_dimensions*[1]               |
| | *short*                         | |                                           |
| | Number of ranges                | |                                           | 
+-----------------------------------+---------------------------------------------+
| | **frang**                       | | *first_range*                             |
| | *short*                         | |                                           |
| | Distance to first range         | |                                           |
| | (kilometers)                    | |                                           |
+-----------------------------------+---------------------------------------------+
| | **rsep**                        | | *range_sep*                               |
| | *short*                         | |                                           |
| | Range separation (kilometers)   | |                                           |
+-----------------------------------+---------------------------------------------+
| | **xcf**                         | | If *xcfs* exist, then =1                  |
| | *short*                         | |                                           |
| | XCF flag                        | |                                           |
+-----------------------------------+---------------------------------------------+
| | **tfreq**                       | | *freq*                                    |
| | *short*                         | |                                           |
| | Transmitted frequency           | |                                           |
+-----------------------------------+---------------------------------------------+
| | **mxpwr**                       | | = -1 (filler)                             |
| | *int*                           | |                                           |
| | Maximum power (kHz)             | |                                           |
+-----------------------------------+---------------------------------------------+
| | **lvmax**                       | | = 20000 (filler)                          |
| | *int*                           | |                                           |
| | Maximum noise level allowed     | |                                           |
+-----------------------------------+---------------------------------------------+
| | **rawacf.revision.major**       | | = 255                                     |
| | *int*                           | |                                           |
| | Major version number of the     | |                                           |
| | rawacf format                   | |                                           |
+-----------------------------------+---------------------------------------------+
| | **rawacf.revision.minor**       | | = 255                                     |
| | *int*                           | |                                           |
| | Minor version number of the     | |                                           |
| | rawacf format                   | |                                           |
+-----------------------------------+---------------------------------------------+
| | **combf**                       | | Original Borealis filename, ‘converted    |
| | *string*                        | | from Borealis file beam number ’ X,       |
| | Comment buffer                  | | number of beams in this original record   | 
| | Comment buffer                  | | (len(beam_nums)), experiment_comment and  |
| |                                 | | slice_comment from the file               |
+-----------------------------------+---------------------------------------------+
| | **thr**                         | | = 0.0 (filler)                            | 
| | *float*                         | |                                           |      
| | Thresholding factor             | |                                           |
+-----------------------------------+---------------------------------------------+
| | **ptab[mppul]**                 | | pulses                                    |
| | *short*                         | |                                           |
| | Pulse table                     | |                                           |
+-----------------------------------+---------------------------------------------+
| | **ltab[2][mplgs]**              | | np.transpose(*lags*)                      |
| | *short*                         | |                                           |
| | Lag table                       | |                                           |
+-----------------------------------+---------------------------------------------+
| | **pwr0[nrang]**                 | | Calculated from *main_acfs*               | 
| | *[float]*                       | |                                           |
| | Lag zero power for main         | |                                           |
+-----------------------------------+---------------------------------------------+
| | **slist[0-nrang]**              | | range(0,*correlation_dimensions*.size[1]) |
| | *[short]*                       | |                                           |
| | List of stored ranges, length   | |                                           |
| | dependent on SNR. Lists the     | |                                           |
| | range gate of each stored ACF   | |                                           |
+-----------------------------------+---------------------------------------------+
| | **acfd[2][mplgs][0-nrang]**     | | *main_acfs* conversion, real and imag     |
| | *[short]*                       | |                                           |
| | Calculated ACFs                 | |                                           |
+-----------------------------------+---------------------------------------------+
| | **xcfd[2][mplgs][0-nrang]**     | | *xcfs* conversion, real and imag          |
| | *[short]*                       | |                                           |
| | Calculated XCFs                 | |                                           |
+-----------------------------------+---------------------------------------------+

If blanked\_samples != ptab, or pulse\_phase\_offset contains non-zeroes, no conversion to dmap rawacf is possible.
