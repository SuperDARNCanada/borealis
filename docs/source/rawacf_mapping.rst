RAWACF DARN FIELDS

| **Var name** | **type** | **description** | **Borealis conversion** |
| --- | --- | --- | --- |
| radar.revision.major | char | Major version number | Borealis major version number, or 255 if not a commit with a version tag |
| radar.revision.minor | char | Minor version number | Borealis minor version number, or = 255 if not a commit with a version tag |
| origin.code | char | Code indicating origin of data | = 100, this can be used as a flag that the origin code was Borealis |
| origin.time | string | ASCII representation of when the data was generated | Timestamp\_of\_write conversion |
| origin.command | string | The command line or control program used to generate the data | &#39;Borealis vXXX&#39; + borealis\_git\_hash + experiment\_name |
| cp | short | Control program identifier | Experiment\_id, truncated to short. Experiment\_id will be some sort of hash of experiment\_name |
| stid | short | Station identifier | Station conversion |
| time.yr | short | Year | Sqn\_timestamps[0] conversion |
| time.mo | short | Month | Sqn\_timestamps[0] conversion |
| time.dy | short | Day | Sqn\_timestamps[0] conversion |
| time.hr | short | Hour | Sqn\_timestamps[0] conversion |
| time.mt | short | Minute | Sqn\_timestamps[0] conversion |
| time.sc | short | Second | Sqn\_timestamps[0] conversion |
| time.us | short | Microsecond | Sqn\_timestamps[0] conversion |
| txpow | short | Transmitted power (kW) | = -1 (filler) |
| nave | shor | Number of pulse sequences transmitted | num\_sequences |
| atten | short | Attenuation level | = 0 (filler) |
| lagfr | short | Lag to first range (microseconds) | first\_range\_rtt |
| smsep | short | Sample separation (microseconds) | (rx\_sample\_rate)^ -1  |
| ercod | short | Error code | = 0 (filler) |
| stat.agc | short | AGC status word | = 0 (filler) |
| stat.lopwr | short | LOPWR status word | = 0 (filler) |
| noise.search | float | Calculated noise from clear frequency search | Noise\_at\_freq[0] conversion |
| noise.mean | float | Average noise across frequency band | = 0 (filler) |
| channel | short | Channel number for a stereo radar (zero for all others) | Slice\_id \*\*\* documentation to be written on what a slice\_id |
| bmnum | short | Beam number | beam\_nums[i] |
| bmazm | float | Beam azimuth | beam\_azms[i] |
| scan | short | Scan flag | Scan\_start\_marker (0 or 1) |
| offset | short | Offset between channels for a  stereo radar (zero for all others) | = 0 (filler) |
| rxrise | short | Receiver rise time (microseconds) | = 0.0 |
| intt.sc | short | Whole number of seconds of integration time. | Int\_time conversion |
| intt.us | short | Fractional number of microseconds of integration time | Int\_time conversion |
| txpl | short | Transmit pulse length (microseconds) | tx\_pulse\_len |
| mpinc | short | Multi-pulse increment (microseconds) | tau\_spacing |
| mppul | short | Number of pulses in sequence | len(pulses) |
| mplgs | short | Number of lags in sequence | lags.shape[0] |
| nrang | short | Number of ranges | correlation\_dimensions[1] |
| frang | short | Distance to first range (kilometers) | first\_range |
| rsep | short | Range separation (kilometers) | range\_sep |
| xcf | short | XCF flag | If &#39;xcfs&#39; exist, then =1 |
| tfreq | short | Transmitted frequency | freq |
| mxpwr | int | Maximum power (kHz) | = -1 (filler) |
| lvmax | int | Maximum noise level allowed | = 20000 (filler) |
| rawacf.revision.major | int | Major version number of the rawacf format |= 255 |
| rawacf.revision.minor | int | Minor version number of the rawacf format |= 255 |
| combf | string | Comment buffer | Original Borealis filename, &#39;converted from Borealis file beam number &#39; , number of beams in this original record (len(beam\_nums)) Original comment\_buffer from file |
| thr | float | Thresholding factor | = 0.0 (filler) |
| ptab[mppul] | [short] | Pulse table | pulses |
| ltab[2][mplgs] | [short] | Lag table | np.transpose(lags) |
| pwr0[nrang] | [float] | Lag zero power for main | Calculated from main\_acfs |
| slist[0-nrang] | [short] | List of stored ranges, length dependent on SNR. Lists the range gate of each stored ACF | range(0,correlation\_dimensions.size[1]) |
| acfd[2][mplgs[0-nrang] | [short] | Calculated ACFs | Main\_acfs conversion, real and imag |
| xcfd[2][mplgs][0-nrang] | [short] | Calculated XCFs | Xcfs conversion, real and imag |


If blanked\_samples != ptab, or pulse\_phase\_offset contains non-zeroes, no conversion to dmap rawacf is possible.
