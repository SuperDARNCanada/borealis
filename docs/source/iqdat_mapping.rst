=============
iqdat_mapping
=============

+-----------------------+----------+--------------------------------+--------------------------------+
| Variable name         | Type     | Description                    | Borealis conversion            |
+=======================+==========+================================+================================+
| radar.revision.major  | char     | Major version number           | Borealis major version number, |
|                       |          |                                | or 255 if not a commit with a  |
|                       |          |                                | version tag                    |
+-----------------------+----------+--------------------------------+--------------------------------+
| radar.revision.minor  | char     | Minor version number           | Borealis minor version number, |
|                       |          |                                | or = 255 if not a commit with  |
|                       |          |                                | a version tag                  |
+-----------------------+----------+--------------------------------+--------------------------------+
| origin.code           | char     | Code indicating origin of data | = 100, this can be used as a   |
|                       |          |                                | flag that the origin code was  |
|                       |          |                                | Borealis                       |
+-----------------------+----------+--------------------------------+--------------------------------+
| origin.time           | string   | ASCII representation of when   | Timestamp\_of\_write           |
|                       |          | the data was generated         | conversion                     |
+-----------------------+----------+--------------------------------+--------------------------------+
| origin.command        | string   | The command line or control    | &#39;Borealis vXXX&#39; +      |
|                       |          | program used to generate the   | borealis\_git\_hash +          |
|                       |          | data                           | experiment\_name               |
+-----------------------+----------+--------------------------------+--------------------------------+
| cp                    | short    | Control program identifier     | Experiment\_id, truncated to   |
|                       |          |                                | short. Experiment\_id will be  |
|                       |          |                                | some sort of hash of           |
|                       |          |                                | experiment\_name               |
+-----------------------+----------+--------------------------------+--------------------------------+
| stid                  | short    | Station identifier             | Station conversion             |
+-----------------------+----------+--------------------------------+--------------------------------+
| time.yr               | short    | Year                           | Sqn\_timestamps[0] conversion  |
+-----------------------+----------+--------------------------------+--------------------------------+
| time.mo               | short    | Month                          | Sqn\_timestamps[0] conversion  |
+-----------------------+----------+--------------------------------+--------------------------------+
| time.dy               | short    | Day                            | Sqn\_timestamps[0] conversion  |
+-----------------------+----------+--------------------------------+--------------------------------+
| time.hr               | short    | Hour                           | Sqn\_timestamps[0] conversion  |
+-----------------------+----------+--------------------------------+--------------------------------+
| time.mt               | short    | Minute                         | Sqn\_timestamps[0] conversion  |
+-----------------------+----------+--------------------------------+--------------------------------+
| time.sc               | short    | Second                         | Sqn\_timestamps[0] conversion  |
+-----------------------+----------+--------------------------------+--------------------------------+
| time.us               | short    | Microsecond                    | Sqn\_timestamps[0] conversion  |
+-----------------------+----------+--------------------------------+--------------------------------+
| txpow                 | short    | Transmitted power (kW)         | = -1 (filler)                  |
+-----------------------+----------+--------------------------------+--------------------------------+
| nave                  | short    | Number of pulse sequences      | num\_sequences                 |
|                       |          | transmitted                    |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| atten                 | short    | Attenuation level              | = 0 (filler)                   |
+-----------------------+----------+--------------------------------+--------------------------------+
| lagfr                 | short    | Lag to first range             | first\_range\_rtt              |
|                       |          | (microseconds)                 |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| smsep                 | short    | Sample separation              | (rx\_sample\_rate)^ -1         |
|                       |          | (microseconds)                 |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| ercod                 | short    | Error code                     | = 0 (filler)                   |
+-----------------------+----------+--------------------------------+--------------------------------+
| stat.agc              | short    | AGC status word                | = 0 (filler)                   |
+-----------------------+----------+--------------------------------+--------------------------------+
| stat.lopwr            | short    | LOPWR status word              | = 0 (filler)                   |
+-----------------------+----------+--------------------------------+--------------------------------+
| noise.search          | float    | Calculated noise from clear    | Noise\_at\_freq[0] conversion  |
|                       |          | frequency search               |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| noise.mean            | float    | Average noise across frequency | = 0 (filler)                   |
|                       |          | band                           |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| channel               | short    | Channel number for a stereo    | Slice\_id \*\*\* documentation |
|                       |          | radar (zero for all others)    | to be written on what a        |
|                       |          |                                | slice\_id                      |
+-----------------------+----------+--------------------------------+--------------------------------+
| bmnum                 | short    | Beam number                    | beam\_nums[i]                  |
+-----------------------+----------+--------------------------------+--------------------------------+
| bmazm                 | float    | Beam azimuth                   | beam\_azms[i]                  |
+-----------------------+----------+--------------------------------+--------------------------------+
| scan                  | short    | Scan flag                      | Scan\_start\_marker (0 or 1)   |
+-----------------------+----------+--------------------------------+--------------------------------+
| offset                | short    | Offset between channels for a  |                                |
|                       |          | stereo radar (zero for all     |                                |
|                       |          | others)                        |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| rxrise                | short    | Receiver rise time             | = 0.0                          |
|                       |          | (microseconds)                 |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| intt.sc               | short    | Whole number of seconds of     | Int\_time conversion           |
|                       |          | integration time.              |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| intt.us               | short    | Fractional number of           | Int\_time conversion           |
|                       |          | microseconds of integration    |                                |
|                       |          | time                           |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| txpl                  | short    | Transmit pulse length          | tx\_pulse\_len                 |
|                       |          | (microseconds)                 |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| mpinc                 | short    | Multi-pulse increment          | tau\_spacing                   |
|                       |          | (microseconds)                 |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| mppul                 | short    | Number of pulses in sequence   | len(pulses)                    |
+-----------------------+----------+--------------------------------+--------------------------------+
| mplgs                 | short    | Number of lags in sequence     | lags.shape[0]                  |
+-----------------------+----------+--------------------------------+--------------------------------+
| nrang                 | short    | Number of ranges               | num\_ranges                    |
+-----------------------+----------+--------------------------------+--------------------------------+
| frang                 | short    | Distance to first range        | first\_range                   |
|                       |          | (kilometers)                   |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| rsep                  | short    | Range separation (kilometers)  | range\_sep                     |
+-----------------------+----------+--------------------------------+--------------------------------+
| xcf                   | short    | XCF flag                       | If &#39;intf&#39;              |
|                       |          |                                | in antenna\_arrays\_order,     |
|                       |          |                                | then xcf = 1 (xcfs are         |
|                       |          |                                | possible)                      |
+-----------------------+----------+--------------------------------+--------------------------------+
| tfreq                 | short    | Transmitted frequency          | freq                           |
+-----------------------+----------+--------------------------------+--------------------------------+
| mxpwr                 | int      | Maximum power (kHz)            | = -1 (filler)                  |
+-----------------------+----------+--------------------------------+--------------------------------+
| lvmax                 | int      | Maximum noise level allowed    | = 20000 (filler)               |
+-----------------------+----------+--------------------------------+--------------------------------+
| iqdata.revision.major | int      | Major version number of the    | = 1 (meaning Borealis          |
|                       |          | iqdata library                 | conversion)                    |
+-----------------------+----------+--------------------------------+--------------------------------+
| iqdata.revision.minor | int      | Minor version number of the    | = 0 (Borealis conversion)      |
|                       |          | iqdata library                 |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| combf                 | string   | Comment buffer                 | Original Borealis filename,    |
|                       |          |                                | &#39;converted from Borealis   |
|                       |          |                                | file &#39; , number of beams   |
|                       |          |                                | in this original record        |
|                       |          |                                | (len(beam\_nums)),             |
|                       |          |                                | experiment\_comment and        |
|                       |          |                                | slice\_comment from the file   |
+-----------------------+----------+--------------------------------+--------------------------------+
| seqnum                | int      | Number of pulse sequences      | num\_sequences                 |
|                       |          | transmitted                    |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| chnnum                | int      | Number of channels sampled     | len(antenna\_arrays\_order)    |
|                       |          | (both I and Q quadrature       |                                |
|                       |          | samples)                       |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| smpnum                | int      | Number of samples taken per    | num\_samps                     |
|                       |          | sequence                       |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| skpnum                | int      | Number of samples to skip      | math.ceil(first\_range/        |
|                       |          | before the first valid sample  | range\_sep). In theory this    |
|                       |          |                                | should =0 due to Borealis      |
|                       |          |                                | functionality (no rise time to |
|                       |          |                                | account for). However          |
|                       |          |                                | make\_raw in RST requires      |
|                       |          |                                | this to be                     |
|                       |          |                                | indicative of the first range  |
|                       |          |                                | so we provide this.            |
+-----------------------+----------+--------------------------------+--------------------------------+
| ptab[mppul]           | short    | Pulse table                    | pulses                         |
+-----------------------+----------+--------------------------------+--------------------------------+
| ltab[2][mplgs]        | short    | Lag table                      | np.transpose(lags)             |
+-----------------------+----------+--------------------------------+--------------------------------+
| tsc[seqnum]           | int      | Seconds component of time past | Sqn\_timestamps conversion     |
|                       |          | epoch of pulse sequence        |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| tus[seqnum]           | int      | Microsecond component of time  | Sqn\_timestamps conversion     |
|                       |          | past epoch of pulse sequence   |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| tatten[seqnum]        | short    | Attenuator setting for each    | = [0,0…] (fillers)             |
|                       |          | pulse sequence                 |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| tnoise[seqnum]        | float    | Noise value for each pulse     | Noise\_at\_freq conversion     |
|                       |          | sequence                       |                                |
+-----------------------+----------+--------------------------------+--------------------------------+
| toff[seqnum]          | int      | Offset into the sample buffer  | Offset = 2 \* num\_samps \*    |
|                       |          | for each pulse sequence        | len(antenna\_arrays\_order)    |
|                       |          |                                | toff = [i \* offset for i in   |
|                       |          |                                | range(                         |
|                       |          |                                | v[&#39;num\_sequences&#39;])]  |
+-----------------------+----------+--------------------------------+--------------------------------+
| tsze[seqnum]          | int      | Number of words stored for     |                                |
|                       |          | this pulse sequence            | = [offset, offset, offset….]   |
+-----------------------+----------+--------------------------------+--------------------------------+
| data[totnum]          | int      | Array of raw I and Q samples,  | Data conversion for correct    |
|                       |          | arranged: [[[smpnum(i),        | dimensions                     |
|                       |          | smpnum(q)] \* chnnum] \*       |                                |
|                       |          | seqnum], so totnum = 2\*seqnum |                                |
|                       |          | \*chnnum\*smpnum               |                                |
+-----------------------+----------+--------------------------------+--------------------------------+

If blanked\_samples != ptab, or pulse\_phase\_offset contains non-zeroes, no conversion to iqdat is possible.
