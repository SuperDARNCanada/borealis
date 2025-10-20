.. _antennas_iq:

-----------
antennas_iq
-----------

    Each field contains metadata that determines how the field is written to file.

    ``description``
       A description of the field.

    ``dim_labels``
       If applicable, a brief descriptor for each dimension of the dataset. This could
       be different for different ``group`` values. If so, this metadata will be a dict, with the keys
       being the ``group`` name and the values the associated list of dimension labels.

    ``dim_scales``
       If applicable, dimension scales will be associated to the field. These are datasets that match
       one of the dimensions of the data, such as timestamps to go along with an array of collected data. Note that
       some dimensions may be associated with multiple fields. If a dimension has no associated dataset, the list
       will have a ``None`` entry.

    ``groups``
       The types of data file that need this field to be written.

    ``level``
       The level within the file that the data will be stored at. Either ``file`` or ``record``, indicating
       that the field is either written once per file, or once per record.

    ``nickname``
       A nickname for the field, used for making Dimension Scale names.

    ``units``
       Units for the data.


Fields
------
:agc_status_word: 32 bits, a 1 in bit position corresponds to an AGC fault on that transmitter

   * level: ``record``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:antenna_locations: Relative antenna locations

   * dim_labels: ``['antenna', 'local_coord']``
   * dim_scales: ``['antennas', 'local_coord']``
   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``
   * units: ``m``


:antennas: Labels for each antenna of the radar

   * dim_labels: ``['antenna']``
   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:antennas_iq_data: Filtered and downsampled I&Q complex voltage samples for each antenna

   * dim_labels: ``['antenna', 'sequence', 'time']``
   * dim_scales: ``['rx_antennas', 'sqn_timestamps', 'sample_time']``
   * level: ``record``
   * required_for: ``['antennas_iq']``
   * units: ``a.u. ~ V``


:beam_azms: Beams azimuths for each beam in degrees CW of boresight

   * dim_labels: ``['beam']``
   * level: ``record``
   * nickname: ``beam direction``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf']``
   * units: ``degrees``


:beam_nums: Beam numbers used in this slice

   * dim_labels: ``['beam']``
   * level: ``record``
   * nickname: ``beam number``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf']``


:blanked_samples: Samples blanked during transmission of a pulse

   * dim_labels: ``['time']``
   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:borealis_git_hash: Version and commit hash of Borealis at runtime

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:cfs_freqs: Frequencies measured during clear frequency search

   * dim_labels: ``['freq']``
   * level: ``file``
   * nickname: ``freq``
   * required_for: ``[]``
   * units: ``Hz``


:cfs_masks: Mask for cfs_freqs restricting freqs available for setting cfs slice freq

   * dim_labels: ``['freq']``
   * dim_scales: ``['cfs_freqs']``
   * level: ``record``
   * required_for: ``[]``


:cfs_noise: Power measured during clear frequency search

   * dim_labels: ``['freq']``
   * dim_scales: ``['cfs_freqs']``
   * level: ``record``
   * required_for: ``[]``
   * units: ``a.u. ~ dBW``


:cfs_range: Lower and upper bound of frequencies examined by clear frequency search

   * level: ``file``
   * required_for: ``[]``
   * units: ``Hz``


:data_normalization_factor: Cumulative scale of all of the filters for a total scaling factor to normalize by

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf']``


:experiment_comment: Comment about the whole experiment

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:experiment_id: Number used to identify experiment

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:experiment_name: Name of the experiment class

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:first_range: Distance to first range in km

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf']``
   * units: ``km``


:first_range_rtt: Round trip time of flight to first range in microseconds

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf']``
   * units: ``μs``


:freq: Frequency used for this experiment slice, in kHz

   * level: ``record``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``
   * units: ``kHz``


:global_coord: Descriptors for global coordinates

   * dim_labels: ``['global_coord']``
   * level: ``file``
   * nickname: ``global coord``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:gps_locked: True if the GPS was locked during the entire averaging period

   * level: ``record``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:gps_to_system_time_diff: Max time diff in seconds between GPS and system/NTP time during the averaging period

   * level: ``record``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``
   * units: ``s``


:int_time: Integration time in seconds

   * level: ``record``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``
   * units: ``s``


:lags: Lag indices

   * level: ``file``
   * nickname: ``lag``
   * required_for: ``['rawacf']``


:lag_numbers: Difference in units of tau_spacing of unique pairs of pulse in the pulse array

   * level: ``file``
   * nickname: ``lag``
   * required_for: ``['rawacf']``
   * units: ``tau_spacing``


:lag_pulses: Unique pairs of pulses in pulse array, in units of tau_spacing

   * dim_labels: ``['lag', 'pulse']``
   * dim_scales: ``['lags', 'lag_pulse_descriptors']``
   * level: ``file``
   * required_for: ``['rawacf']``
   * units: ``tau_spacing``


:lag_pulse_descriptors: Descriptor of the pulse pairs used in a lag

   * level: ``file``
   * required_for: ``['rawacf']``


:local_coord: Descriptors for local coordinates

   * dim_labels: ``['local_coord']``
   * level: ``file``
   * nickname: ``local coord``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:lp_status_word: 32 bits, a 1 in bit position corresponds to a low power condition on that transmitter

   * level: ``record``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:num_sequences: Number of sampling periods in the averaging period

   * level: ``record``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:num_slices: Number of slices in the experiment for this averaging period

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:pulse_phase_offset: Phase offset in degrees for each pulse in pulses

   * dim_labels: ``['sequence', 'pulse']``
   * dim_scales: ``['sqn_timestamps', 'pulses']``
   * level: ``record``
   * required_for: ``[]``
   * units: ``degrees``


:pulses: Pulse sequence in units of tau_spacing

   * dim_labels: ``['pulse']``
   * level: ``file``
   * nickname: ``pulse``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf']``


:range_gates: Range gates of interest for the experiment, beginning at ``first_range`` and spaced by ``range_sep``

   * level: ``file``
   * nickname: ``range gate``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf']``


:range_sep: Range gate separation (equivalent distance between samples) in km

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf']``
   * units: ``km``


:rx_antennas: Indices into ``antenna_locations`` of the antennas with recorded data

   * level: ``file``
   * nickname: ``rx antenna``
   * required_for: ``['antennas_iq', 'rawrf']``


:rx_intf_antennas: Indices into ``antenna_locations`` of the interferometer array antennas used in this experiment

   * level: ``file``
   * required_for: ``[]``


:rx_main_antennas: Indices into ``antenna_locations`` of the main array antennas used in this experiment

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:rx_sample_rate: Sampling rate of the samples being written to file in Hz

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``
   * units: ``Hz``


:rx_main_excitations: Complex excitations of main array receive antennas for each antenna. Magnitude between 0 (off) and 1 (full power)

   * dim_labels: ``['beam', 'antenna']``
   * dim_scales: ``[['beam_azms', 'beam_nums'], 'rx_main_antennas']``
   * level: ``record``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:rx_intf_excitations: Complex excitations of interferometer array receive antennas for each antenna. Magnitude between 0 (off) and 1 (full power)

   * dim_labels: ``['beam', 'antenna']``
   * dim_scales: ``[['beam_azms', 'beam_nums'], 'rx_intf_antennas']``
   * level: ``record``
   * required_for: ``[]``


:samples_data_type: C data type of the samples

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:sample_time: Time of measurement relative to the first pulse in the sequence

   * dim_labels: ``['time']``
   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawrf']``
   * units: ``μs``


:scan_start_marker: Designates if the record is the first in a scan

   * level: ``record``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:scheduling_mode: Type of scheduling time at the time of this dataset

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:slice_comment: Comment that describes the slice

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf']``


:slice_id: Slice ID of the file and dataset

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf']``


:slice_interfacing: Interfacing of this slice to other slices

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf']``


:sqn_timestamps: GPS timestamps of start of first pulse for each sampling period in the averaging period

   * dim_labels: ``['sequence']``
   * level: ``record``
   * nickname: ``timestamp``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``
   * units: ``seconds since 1970-01-01 00:00:00 UTC``


:station: Three letter radar identifier

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:station_location: Location of the radar

   * dim_labels: ``['global_coord']``
   * dim_scales: ``['global_coord']``
   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:tau_spacing: Unit of spacing between pulses in microseconds

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf']``
   * units: ``μs``


:tx_antennas: Indices into ``antenna_locations`` of the antennas used for transmitting in this experiment

   * level: ``file``
   * nickname: ``tx antenna``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf', 'rawrf']``


:tx_excitations: Complex excitations of transmit signal for each antenna. Magnitude between 0 (off) and 1 (full power)

   * dim_labels: ``['antenna']``
   * dim_scales: ``['tx_antennas']``
   * level: ``record``
   * required_for: ``[]``
   * units: ``a.u.``


:tx_pulse_len: Length of the pulse in microseconds

   * level: ``file``
   * required_for: ``['antennas_iq', 'bfiq', 'rawacf']``
   * units: ``μs``
