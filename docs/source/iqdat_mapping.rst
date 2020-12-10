=============
iqdat_mapping
=============

IQDAT SDARN FIELDS

This conversion is done in pyDARNio here in the __convert_bfiq_record method: `Link to Source <https://github.com/SuperDARN/pyDARNio/blob/master/pydarnio/borealis/borealis_convert.py>`_

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
| | **nrang**                       | | *num_ranges*                              |
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
| | **iqdata.revision.major**       | | = 1 (meaning Borealis conversion)         |
| | *int*                           | |                                           |
| | Major version number of the     | |                                           |
| | iqdata library                  | |                                           |
+-----------------------------------+---------------------------------------------+
| | **iqdata.revision.minor**       | | = 0 (Borealis conversion)                 |
| | *int*                           | |                                           |
| | Minor version number of the     | |                                           |
| | iqdata library                  | |                                           |
+-----------------------------------+---------------------------------------------+
| | **combf**                       | | Original Borealis filename, ‘converted    |
| | *string*                        | | from Borealis file ’ , number of beams in | 
| | Comment buffer                  | | this original record (len(beam_nums)),    |
| |                                 | | experiment_comment and slice_comment      |
| |                                 | | from the file                             |
+-----------------------------------+---------------------------------------------+
| | **seqnum**                      | | *num_sequences*                           |
| | *int*                           | |                                           |
| | Number of pulse sequences       | |                                           |
| | transmitted                     | |                                           |
+-----------------------------------+---------------------------------------------+
| | **chnnum**                      | | len(*antenna_arrays_order*)               |
| | *int*                           | |                                           |
| | Number of channels sampled      | |                                           |
| | (both I and Q quadrature        | |                                           |
| | samples)                        | |                                           |
+-----------------------------------+---------------------------------------------+
| | **smpnum**                      | | *num_samps*                               |
| | *int*                           | |                                           |
| | Number of samples taken per     | |                                           |
| | sequence                        | |                                           |
+-----------------------------------+---------------------------------------------+
| | **skpnum**                      | | math.ceil(first_range/range_sep). In      |
| | *int*                           | | theory this should =0 due to Borealis     |
| | Number of samples to skip       | | functionality(no rise time).              | 
| | before the first valid sample   | | However make_raw in RST requires this to  |
| |                                 | | be indicative of the first range so we    |
| |                                 | | provide this.                             |
+-----------------------------------+---------------------------------------------+
| | **ptab[mppul]**                 | | pulses                                    |
| | *short*                         | |                                           |
| | Pulse table                     | |                                           |
+-----------------------------------+---------------------------------------------+
| | **ltab[2][mplgs]**              | | np.transpose(*lags*)                      |
| | *short*                         | |                                           |
| | Lag table                       | |                                           |
+-----------------------------------+---------------------------------------------+
| | **tsc[seqnum]**                 | | *sqn_timestamps* conversion               |
| | *int*                           | |                                           |
| | Seconds component of time past  | |                                           |
| | epoch of pulse sequence         | |                                           |
+-----------------------------------+---------------------------------------------+
| | **tus[seqnum]**                 | | *sqn_timestamps* conversion               |
| | *int*                           | |                                           |
| | Microsecond component of time   | |                                           |
| | past epoch of pulse sequence    | |                                           |
+-----------------------------------+---------------------------------------------+
| | **tatten[seqnum]**              | | = [0,0…] (fillers)                        |
| | *short*                         | |                                           |
| | Attenuator setting for each     | |                                           |
| | pulse sequence                  | |                                           |
+-----------------------------------+---------------------------------------------+
| | **tnoise[seqnum]**              | | *noise_at_freq* conversion                |
| | *float*                         | |                                           |
| | Noise value for each pulse      | |                                           |
| | sequence                        | |                                           |
+-----------------------------------+---------------------------------------------+
| | **toff[seqnum]**                | | Offset = 2 * num_samps *                  |
| | *int*                           | | len(antenna_arrays_order), toff = [i *    |
| | Offset into the sample buffer   | | offset for i in range(v['num_sequences'])]|
| | for each pulse sequence         | |                                           |
+-----------------------------------+---------------------------------------------+
| | **tsze[seqnum]**                | | = [offset, offset, offset….]              | 
| | *int*                           | |                                           |
| | Number of words stored for this | |                                           |
| | pulse sequence                  | |                                           |
+-----------------------------------+---------------------------------------------+
| | **data[totnum]**                | | Data conversion for correct dimensions    |
| | *int*                           | | and scaled to max int (-32768 to 32767)   |
| | Array of raw I and Q samples,   | |                                           |
| | arranged: [[[smpnum(i),         | |                                           |
| | smpnum(q)] * chnnum] * seqnum], | |                                           |
| | so totnum =                     | |                                           |
| | 2*seqnum*chnnum*smpnum          | |                                           |
+-----------------------------------+---------------------------------------------+

If *blanked_samples* != *ptab*, or *pulse_phase_offset* contains non-zeroes, no conversion to iqdat is possible.
