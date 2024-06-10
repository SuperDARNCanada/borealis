#!/usr/bin/env python3

"""
    decimation_scheme
    ~~~~~~~~~~~~~~~~~
    This file contains classes and functions for building decimation schemes

    :copyright: 2018 SuperDARN Canada
"""
# built-in
import math

# third-party
from scipy.signal import firwin, remez, kaiserord


class DecimationStage(object):

    def __init__(self, stage_num, input_rate, dm_rate, filter_taps):
        """
        Create a decimation stage with given decimation rate, input sample rate, and filter taps.

        :param  stage_num:      the index of this filter/decimate stage to all stages (beginning with stage 0)
        :type   stage_num:      int
        :param  input_rate:     the input sampling rate, in Hz.
        :type   input_rate:     float
        :param  dm_rate:        the decimation rate. Must be an integer.
        :type   dm_rate:        int
        :param  filter_taps:    a list of filter taps (numeric) to be convolved with the data before
                                the decimation is done.
        :type   filter_taps:    list

        :raises ExperimentException: if types are not correct for signal processing module to use
        """
        self.stage_num = stage_num
        self.input_rate = input_rate
        if not isinstance(dm_rate, int):
            raise ValueError("Decimation rate is not an integer")
        self.output_rate = input_rate / dm_rate
        self.dm_rate = dm_rate
        if not isinstance(filter_taps, list):
            errmsg = (
                f"Filter taps {filter_taps} of type {type(filter_taps)} must be a list in"
                f" decimation stage {stage_num}"
            )
            raise ValueError(errmsg)
        for x in filter_taps:
            if not isinstance(x, (int, float)):  # TODO should complex be included here?
                errmsg = (
                    f"Filter tap {x} is not numeric in decimation stage {stage_num}"
                )
                raise ValueError(errmsg)
        self.filter_taps = filter_taps

    def __eq__(self, other):
        for k, v in self.__dict__.items():
            if v != getattr(other, k, None):
                return False
        return True


class DecimationScheme(object):
    """
    Class for DSP filtering and decimation scheme.
    """

    def __init__(self, rxrate, output_sample_rate, stages=None):
        """
        Set up the decimation scheme for the experiment.

        :param  rxrate:             sampling rate of USRP, in Hz.
        :type   rxrate:             float
        :param  output_sample_rate: desired output rate of the data, to decimate to, in Hz.
        :type   output_sample_rate: float
        :param  stages:             a list of DecimationStages. Defaults to None
        :type   stages:             list or None
        """
        self.stages = stages
        if stages is None:
            # create the default filters according to default scheme. Currently only creating
            # default filters if sampling rate and output rate are set up as per original design.
            # TODO: make this more general.
            if rxrate != 5.0e6 or round(output_sample_rate, 0) != 3.333e3:
                errmsg = (
                    f"Default filters not defined for rxrate {rxrate} and output rate"
                    f" {output_sample_rate}"
                )
                raise ValueError(errmsg)

            # set up defaults as per design.
            scheme = create_default_scheme()
            self.stages = scheme.stages

        self.rxrate = rxrate
        self.output_sample_rate = output_sample_rate
        self.dm_rates = []
        self.output_rates = []
        self.input_rates = []
        self.filter_scaling_factors = []

        for dec_stage in self.stages:
            self.dm_rates.append(dec_stage.dm_rate)
            self.output_rates.append(dec_stage.output_rate)
            self.input_rates.append(dec_stage.input_rate)
            filter_scaling_factor = sum(dec_stage.filter_taps)
            self.filter_scaling_factors.append(filter_scaling_factor)

        # check rates are appropriate given rxrate and output_sample_rate, and
        # check sequentiality of stages, ie output rate transfers to input rate of next stage.
        if self.input_rates[0] != self.rxrate:
            errmsg = (
                f"Decimation stage 0 does not have input rate {self.input_rates[0]}"
                f" equal to USRP sampling rate {self.rxrate}"
            )
            raise ValueError(errmsg)

        for stage_num in range(0, len(self.stages) - 1):
            if not math.isclose(
                self.output_rates[stage_num],
                self.input_rates[stage_num + 1],
                abs_tol=0.001,
            ):
                errmsg = (
                    f"Decimation stage {stage_num} output rate {self.output_rates[stage_num]}"
                    f" does not equal next stage {stage_num + 1} input rate {self.input_rates[stage_num + 1]}"
                )
                raise ValueError(errmsg)

        if not math.isclose(
            self.output_rates[-1], self.output_sample_rate, abs_tol=0.001
        ):
            errmsg = (
                f"Last decimation stage {len(self.stages) - 1} does not have output rate"
                f" {self.output_rates[-1]} equal to requested output data rate {self.output_sample_rate}"
            )
            raise ValueError(errmsg)

    def __repr__(self):
        repr_str = f"Decimation Scheme with {len(self.stages)} stages:\n"
        for stage in self.stages:
            repr_str += f"\nStage {stage.stage_num}:"
            repr_str += f"\nInput Rate: {stage.input_rate} Hz"
            repr_str += f"\nDecimation by: {stage.dm_rate}"
            repr_str += f"\nOutput Rate: {stage.output_rate} Hz"
            repr_str += f"\nNum taps: {len(stage.filter_taps)}"
            # repr_str += '\nFilter Taps: {}\n'.format(stage.filter_taps)
        return repr_str

    def __eq__(self, other):
        for k, v in self.__dict__.items():
            if v != getattr(other, k, None):
                return False
        return True


def create_default_scheme():
    """
    Previously known as create_test_scheme_9 until July 23/2019!
    Create four stages of FIR filters and a decimation scheme. Returns a decimation scheme of type DecimationScheme.
    This filter will have a wider receive bandwidth than the previous.
    Pasha recommends a 10kHz bandwidth for the final stage. I believe there will be aliasing caused by this but
    perhaps the concern is not critical because of the small bandwidth overlapping. I will test this anyways.

    :returns:   a decimation scheme for use in experiment.
    :rtype:     DecimationScheme
    """

    rates = [5.0e6, 500.0e3, 100.0e3, 50.0e3 / 3]
    dm_rates = [10, 5, 6, 5]
    transition_widths = [150.0e3, 40.0e3, 15.0e3, 1.0e3]
    cutoffs = [20.0e3, 10.0e3, 10.0e3, 5.0e3]  # bandwidth is double this
    ripple_dbs = [150.0, 80.0, 35.0, 9.0]
    scaling_factors = [10.0, 100.0, 100.0, 100.0]
    all_stages = []

    for stage in range(0, 4):
        filter_taps = list(
            scaling_factors[stage]
            * create_firwin_filter_by_attenuation(
                rates[stage],
                transition_widths[stage],
                cutoffs[stage],
                ripple_dbs[stage],
            )
        )
        all_stages.append(
            DecimationStage(stage, rates[stage], dm_rates[stage], filter_taps)
        )

    return DecimationScheme(5.0e6, 10.0e3 / 3, stages=all_stages)


def calculate_num_filter_taps(sampling_freq, trans_width, k):
    """
    Calculates the number of filter taps required for the filter, using the sampling rate,
    transition width desired, and a k value which impacts filter performance (3 typical).

    :param      sampling_freq: sampling rate, Hz.
    :type       sampling_freq:  float
    :param      trans_width:    transition bandwidth between cutoff of passband and stopband (Hz)
    :type       trans_width:    float
    :param      k:              value increasing filter performance with increasing value.
    :type       k:              int

    :returns:   num_taps, number of taps for the FIR filter.
    :rtype:     int
    """
    num_taps = int(k * (sampling_freq / trans_width))
    return num_taps


def create_remez_filter(sampling_freq, cutoff_freq, trans_width):
    """
    Creates the default filter for the experiment, if the filter taps are not provided.

    :param      sampling_freq:  sampling rate, Hz.
    :type       sampling_freq:  float
    :param      cutoff_freq:    cutoff frequency, or edge of passband, Hz.
    :type       cutoff_freq:    float
    :param      trans_width:    transition bandwidth between cutoff of passband and stopband (Hz)
    :rtype      trans_width:    float

    :returns:   lowpass filter taps created using the remez method.
    :rtype:     ndarray
    """
    num_taps = calculate_num_filter_taps(sampling_freq, trans_width, 3)
    lpass = remez(
        num_taps,
        [0, cutoff_freq, cutoff_freq + trans_width, 0.5 * sampling_freq],
        [1, 0],
        fs=sampling_freq,
    )

    return lpass


def create_firwin_filter_by_attenuation(
    sample_rate, transition_width, cutoff_hz, ripple_db, window_type="kaiser"
):
    """
    Create a firwin filter.

    :param      sample_rate:        sample rate
    :type       sample_rate:        float
    :param      transition_width:   transition bandwidth between cutoff of passband and stopband (Hz)
    :type       transition_width:   float
    :param      cutoff_hz:          cutoff frequency, in hz
    :type       cutoff_hz:          float
    :param      ripple_db:          The desired attenuation in the stop band, in dB.
    :type       ripple_db:          float
    :param      window_type:        Window for the filter. Defaults to kaiser
    :type       window_type:        str

    :returns:   taps, coefficient of FIR filter
    :rtype:     ndarray
    """
    # The Nyquist rate of the signal, as we have complex sampled data
    nyq_rate = sample_rate

    # The desired width of the transition from pass to stop, relative to the Nyquist rate.
    width_ratio = transition_width / nyq_rate

    # Compute the order and Kaiser parameter for the FIR filter.
    num_taps, beta = kaiserord(ripple_db, width_ratio)

    # Use firwin with a Kaiser window to create a lowpass FIR filter
    if window_type == "kaiser":
        window = ("kaiser", beta)
    else:
        window = window_type

    taps = firwin(num_taps, 2 * cutoff_hz / nyq_rate, window=window)

    return taps


def create_firwin_filter_by_num_taps(
    sample_rate, cutoff_hz, num_taps, window_type=("kaiser", 8.0)
):
    """
    Create a firwin filter.

    :param      sample_rate:        sample rate
    :type       sample_rate:        float
    :param      cutoff_hz:          cutoff frequency, in hz
    :type       cutoff_hz:          float
    :param      num_taps:           The desired number of filter taps
    :type       num_taps:           int
    :param      window_type:        Window for the filter. Defaults to kaiser
    :type       window_type:        str

    :returns:   taps, coefficient of FIR filter
    :rtype:     ndarray
    """
    # The Nyquist rate of the signal.
    nyq_rate = sample_rate  # because we have complex sampled data.
    taps = firwin(num_taps, 2 * cutoff_hz / nyq_rate, window=window_type)

    return taps
