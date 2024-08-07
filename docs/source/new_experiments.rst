.. _new-experiments:

=============================
New Experimental Capabilities
=============================

.. _full fov imaging:

---------------------------------------
Simultaneous Full Field-of-View Imaging
---------------------------------------

Simultaneous Full Field-of-View (FOV) Imaging is now possible with Borealis! The current
implementation uses phase modulation across the transmitting antennas of the main array. Two
configurations are supported right now: transmission with all 16 main antennas, and transmission
with 8 adjacent antennas of the main array. These two modes are implemented in the experiments
``full_fov.py`` and ``full_fov_2freq.py``, respectively, with the latter using 8 antennas for each of
two frequencies which transmit simultaneously.

To create your own Full FOV experiment using 8- or 16-antenna phase modulation, a utility function
is present in ``superdarn_common_fields.py``. To use this function, set ::

    slice_dict['tx_antenna_pattern'] = scf.easy_widebeam

in your experiment slice dictionary.

More generally, you can define your own power- and phase-modulation across the transmitting
antennas. Define a function with the following signature

.. code-block:: python

    def fn_name(frequency_khz, tx_antennas, antenna_spacing_m):
        """Defines the power and phase modulation for each transmitting antenna.

        Parameters
        ----------
        frequency_khz: int
            Frequency in kHz.
        tx_antennas: list
            List of transmitting antennas for the slice.
        antenna_spacing_m: float
            Spacing between adjacent antennas in the main array, in meters.

        Returns
        -------
        np.ndarray of shape (num_beams, num_main_antennas). Each element should be a complex number
        with magnitude <= 1.0 defining the power and phase for that antenna.
        """

Then, assign the function to ``tx_antenna_pattern`` in your slice dictionary. It is important that
the first dimension of the returned array matches the first dimension of ``rx_beam_order`` in your
slice dictionary, and that ``num_main_antennas`` matches the number of main antennas in the config
file.

Custom beamforming of the results measured during a full field of view experiment is possible
through defining the ``rx_antenna_pattern`` field in the full field of view
experiment. A custom function can be written in the experiment and passed to borealis ::

    beamforming_function(beam_angle, freq, antenna_count, antenna_spacing, offset=0.0):

    ...

    slice_dict['tx_antenna_pattern'] = beamforming_function

The function should expect to receive beam angles, operating frequencies, number of antennas,
antenna spacing, and an offset. This function will be called for both the rx signals from the main
array and the interferometer array. The return is expected to be the desired phase for beamforming
each antenna, and should be of size [beam_angle, antenna_count]. The magnitude of each entry should
be less than or equal to 1.

.. _bistatic experiments:

--------------------
Bistatic Experiments
--------------------

Bistatic experiments are now supported in Borealis. A multi-purpose bistatic experiment is defined
in ``bistatic_test.py``, and can be used at both transmitting and receiving radar sites. This
experiment uses command-line arguments to control its behaviour, making it flexible and
configurable.

By default, the ``bistatic_test`` experiment will transmit a full FOV pattern like ``full_fov.py``
on a single frequency. To operate a radar in a bistatic listening mode, the argument
``listen_to=[three-letter radar code]`` must be passed to the experiment handler via
``steamed_hams.py``. This will look something like this ::

    radar@borealis~$ ./steamed_hams.py bistatic_test release discretionary --kwargs_string "listen_to=rkn"

This invocation will trigger the radar to receive only, tuning to common-mode frequency 1 for RKN
defined in ``superdarn_common_fields.py``. Additionally, it will trigger an imaging mode, receiving
signals on all 16 beams simultaneously. If the listening radar specified is the same as the radar
that you are running the experiment on, the experiment will default to a listening-only mode on the
radar's common-mode frequency 1.

For further control over the transmitting characteristics, an additional keyword argument
``beam_order`` is supported. This controls the ``tx_beam_order`` field of the slice dictionary, and
allows for traditional beams to be used. The ``beam_order`` value must be formatted as a list of
numbers, such as ``0,1,2,3-5,2,9``, with ranges being parsed to include all numbers in between and
both endpoints. Therefore, for this example, beams used would be 0, 1, 2, 3, 4, 5, 2 and finally 9.
Repeated beams are valid.

The arguments ``beam_order`` and ``listen_to`` are mutually exclusive.

You can define your own bistatic experiment, with very few restrictions. It is highly recommended
that the field ``align_sequences`` is set to ``True`` in your experiment slice dictionary, which
will send out the first pulse in each sequence within 1us of each 0.1 second boundary. Without this
field set in the experiments of both radars in the bistatic link, there is no guarantee of timing
synchronicity and the data will likely be useless. Additionally, it is recommended that the
experiments running at both the transmit and receive radars are both using the same scanbound. This
will make it much easier to compare data from the transmit and recieve sites as the averaging
periods should line up exactly. Lastly, it is recommended that you check the data files for both
radars afterwards and ensure that the ``gps_locked`` flag is True for all times. If not, the clock
may have drifted, and the ``sqn_timestamps`` field may be inaccurate.

.. _clear frequency search:

-------------------------------------
Clear Frequency Search (Experimental)
-------------------------------------

An experimental implementation of clear frequency searching is now supported in borealis to determine
a transmit frequency within a specified band. In an experiment slice, the ``freq`` parameter should be
unset, and the ``cfs_range`` parameter set to a two element list containing the upper and lower
frequency limits of the CFS band::

    slice['cfs_range'] = [11000, 11300]  # Lower and upper freq limit in kHz

Multiple experiment slices within an averaging period can be configured to receive a transmit frequency
from the CFS as long as the each slice has ``cfs_range`` set. Each slice can choose any band within the
transmit and receive bandwidth of the system. Be aware when choosing a ``cfs_range`` that if the range
has any part within +/- 50kHz around the ``txctrfreq`` and ``rxctrfreq`` a warning will be raised as
no tx frequency can be chosen that is within 50kHz of the center frequencies. The user should be aware
of any restricted bands within the desired range, as CFS will exclude restricted bands when selecting
transmit frequencies.

Additionally, if a ``cfs_range`` with a band greater than 300kHz is desired, the user will need to
design a custom decimation scheme for the CFS analysis, as the default is designed only for bands of
300kHz or smaller. All CFS slices with **CONCURRENT** or **SEQUENCE** interfacing must have the same
decimation scheme.

The following parameters can be set for a CFS slice:

.. list-table:: Clear Frequency Search Experiment Parameters
   :widths: 25 25 50
   :header-rows: 1

   * - CFS Parameter
     - Default Value
     - Description
   * - ``cfs_range``
     - None
     - Sets the band to search. Must be set to perform CFS
   * - ``cfs_duration``
     - 90 ms
     - Determines how long the CFS sequence will listen for
   * - ``cfs_scheme``
     - ``create_default_cfs_scheme()``
     - Decimation scheme used in analyzing the CFS data. The default scheme is designed for bands
       of 300kHz or less
   * - ``cfs_stable_time``
     - 0 s
     - Sets a minimum amount of time during which CFS will not change the frequency of a CFS slice
       after it has been set
   * - ``cfs_pwr_threshold``
     - ``None`` (dB)
     - Sets a threshold power difference that a CFS scan must exceed before a frequency is switched.
       If another frequency is lower in power than the current frequency was when set by the
       threshold or if the current frequency power has increased by more than the threshold, then
       CFS will set a new frequency for the CFS slices. When set the threshold value must be greater
       than zero.
   * - ``cfs_fft_n``
     - 512
     - Sets the number of samples used in the FFT during CFS processing. Determines the frequency
       resolution of the CFS, where the resolution is ``res = (rx_rate / total decimation rate) / N``

When a CFS slice is to be run during an averaging period, the first sequence of the averaging period
is used to listen for the length of time specified by ``cfs_duration``. The data from this measurement
is analyzed to evaluate the frequency spectrum of the collected data to select the least noisy frequencies
for transmission. The analysis results are also recorded to any generated rawacf, antennas_iq, and/or
bfiq files.
