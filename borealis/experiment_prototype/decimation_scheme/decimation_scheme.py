
import os
import sys
import math
from scipy.signal import firwin, remez, kaiserord

from borealis.utils.options.experimentoptions import ExperimentOptions
from borealis.experiment_prototype.experiment_exception import ExperimentException


class DecimationStage(object):

    def __init__(self, stage_num, input_rate, dm_rate, filter_taps):
        """
        Create a decimation stage with given decimation rate, input sample rate, and filter taps.
        :param stage_num: the index of this filter/decimate stage to all stages (beginning with
        stage 0)
        :param input_rate: the input sampling rate, in Hz.
        :param dm_rate: the decimation rate. Must be an integer.
        :param filter_taps: a list of filter taps (numeric) to be convolved with the data before the
        decimation is done.
        :raises ExperimentException: if types are not correct for signal processing module to use
        """
        self.stage_num = stage_num
        self.input_rate = input_rate
        if not isinstance(dm_rate, int):
            raise ExperimentException('Decimation rate is not an integer')
        self.output_rate = input_rate/dm_rate
        self.dm_rate = dm_rate
        if not isinstance(filter_taps, list):
            errmsg = 'Filter taps {} of type {} must be a list in decimation stage {}'.format(filter_taps, type(filter_taps), stage_num)
            raise ExperimentException(errmsg)
        for x in filter_taps:
            if not isinstance(x, (int, float)):  # TODO should complex be included here?
                errmsg = 'Filter tap {} is not numeric in decimation stage {}'.format(x, stage_num)
                raise ExperimentException(errmsg)
        self.filter_taps = filter_taps


class DecimationScheme(object):
    """
    Class for DSP filtering and decimation scheme.
    """

    def __init__(self, rxrate, output_sample_rate, stages=None):
        """
        Set up the decimation scheme for the experiment.
        :param rxrate: sampling rate of USRP, in Hz.
        :param output_sample_rate: desired output rate of the data, to decimate to, in Hz.
        :param stages: a list of DecimationStages, or None, if they will be set up as default here.
        """

        if stages is None:  # create the default filters according to default scheme.
            # Currently only creating default filters if sampling rate and output rate are set
            # up as per original design. TODO: make this more general.
            if rxrate != 5.0e6 or round(output_sample_rate, 0) != 3.333e3:
                errmsg = 'Default filters not defined for rxrate {} and output rate {}'.format(
                    rxrate, output_sample_rate)
                raise ExperimentException(errmsg)

            # set up defaults as per design.
            return(create_default_scheme())

        else:
            options = ExperimentOptions()
            self.rxrate = rxrate
            self.output_sample_rate = output_sample_rate
            # check that number of stages is correct
            if len(stages) > options.max_number_of_filtering_stages:
                errmsg = 'Number of decimation stages ({}) is greater than max available {}' \
                         ''.format(len(stages), options.max_number_of_filtering_stages)
                raise ExperimentException(errmsg)
            self.dm_rates = []
            self.output_rates = []
            self.input_rates = []
            self.filter_scaling_factors = []

            self.stages = stages
            for dec_stage in self.stages:
                self.dm_rates.append(dec_stage.dm_rate)
                self.output_rates.append(dec_stage.output_rate)
                self.input_rates.append(dec_stage.input_rate)
                filter_scaling_factor = sum(dec_stage.filter_taps)
                self.filter_scaling_factors.append(filter_scaling_factor)

            # check rates are appropriate given rxrate and output_sample_rate, and
            # check sequentiality of stages, ie output rate transfers to input rate of next stage.
            if self.input_rates[0] != self.rxrate:
                errmsg = 'Decimation stage 0 does not have input rate {} equal to USRP sampling ' \
                         'rate {}'.format(self.input_rates[0], self.rxrate)
                raise ExperimentException(errmsg)

            for stage_num in range(0, len(stages) -1):
                if not math.isclose(self.output_rates[stage_num], self.input_rates[stage_num + 1], abs_tol=0.001):
                    errmsg = 'Decimation stage {} output rate {} does not equal next stage {} ' \
                             'input rate {}'.format(stage_num, self.output_rates[stage_num],
                                                    stage_num + 1, self.input_rates[stage_num + 1])
                    raise ExperimentException(errmsg)

            if self.output_rates[-1] != self.output_sample_rate:
                errmsg = 'Last decimation stage {} does not have output rate {} equal to ' \
                         'requested output data rate {}'.format(len(stages) - 1,
                                                                self.output_rates[-1],
                                                                self.output_sample_rate)
                raise ExperimentException(errmsg)

    def __repr__(self):
        repr_str = 'Decimation Scheme with {} stages:\n'.format(len(stages))
        for stage in self.stages:
            repr_str += '\nStage {}:'.format(stage.stage_num)
            repr_str += '\nInput Rate: {} Hz'.format(stage.input_rate)
            repr_str += '\nDecimation by: {}'.format(stage.dm_rate)
            repr_str += '\nOutput Rate: {} Hz'.format(stage.output_rate)
            repr_str += '\nNum taps: {}'.format(len(stage.filter_taps))
            #repr_str += '\nFilter Taps: {}\n'.format(stage.filter_taps)
        return repr_str


def create_default_scheme(): 
    """
    Previously known as create_test_scheme_9 until July 23/2019! 
    Create four stages of FIR filters and a decimation scheme. Returns a decimation scheme of type DecimationScheme. 
    This filter will have a wider receive bandwidth than the previous.
    Pasha recommends a 10kHz bandwidth for the final stage. I believe there will be aliasing caused by this but 
    perhaps the concern is not critical because of the small bandwidth overlapping. I will test this anyways.

    :return DecimationScheme: a decimation scheme for use in experiment.
    """

    rates = [5.0e6, 500.0e3, 100.0e3, 50.0e3/3]
    dm_rates = [10, 5, 6, 5]
    transition_widths = [150.0e3, 40.0e3, 15.0e3, 1.0e3]
    cutoffs = [20.0e3, 10.0e3, 10.0e3, 5.0e3] # bandwidth is double this
    ripple_dbs = [150.0, 80.0, 35.0, 9.0]
    scaling_factors = [10.0, 100.0, 100.0, 100.0]
    all_stages = []

    for stage in range(0,4):
        filter_taps = list(scaling_factors[stage] * create_firwin_filter_by_attenuation(rates[stage], transition_widths[stage], cutoffs[stage], ripple_dbs[stage]))
        all_stages.append(DecimationStage(stage, rates[stage], dm_rates[stage], filter_taps))

    return (DecimationScheme(5.0e6, 10.0e3/3, stages=all_stages))


def calculate_num_filter_taps(sampling_freq, trans_width, k):
    """
    Calculates the number of filter taps required for the filter, using the sampling rate,
    transition width desired, and a k value which impacts filter performance (3 typical).
    :param sampling_freq: sampling rate, Hz.
    :param trans_width: transition bandwidth between cutoff of passband and stopband (Hz)
    :param k: value increasing filter performance with increasing value. Integer.
    :return: num_taps, number of taps for the FIR filter.
    """
    num_taps = int(k * (sampling_freq /trans_width))
    return num_taps


def create_remez_filter(sampling_freq, cutoff_freq, trans_width):
    """
    Creates the default filter for the experiment, if the filter taps are not provided.
    :param sampling_freq: sampling rate, Hz.
    :param cutoff_freq: cutoff frequency, or edge of passband, Hz.
    :param trans_width: transition bandwidth between cutoff of passband and stopband (Hz)
    :return: lowpass filter taps created using the remez method.
    """

    num_taps = calculate_num_filter_taps(sampling_freq, trans_width, 3)

    lpass = remez(num_taps, [0, cutoff_freq, cutoff_freq + trans_width,
                                    0.5 * sampling_freq], [1, 0], Hz=sampling_freq)

    return lpass


def create_firwin_filter_by_attenuation(sample_rate, transition_width, cutoff_hz, ripple_db, 
    window_type='kaiser'):
    """
    Create a firwin filter. 

    :param ripple_db: The desired attenuation in the stop band, in dB.
    """

    # The Nyquist rate of the signal.
    nyq_rate = sample_rate  # because we have complex sampled data. 

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate. '
    width_ratio = transition_width/nyq_rate

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width_ratio)

    # Use firwin with a Kaiser window to create a lowpass FIR filter
    if window_type == 'kaiser':
        window = ('kaiser', beta)
    else:
        window = window_type

    taps = firwin(N, 2*cutoff_hz/nyq_rate, window=window)

    return taps


def create_firwin_filter_by_num_taps(sample_rate, transition_width, cutoff_hz, num_taps, 
    window_type=('kaiser', 8.0)):
    """
    Create a firwin filter. 

    :param ripple_db: The desired attenuation in the stop band, in dB.
    """

    # The Nyquist rate of the signal.
    nyq_rate = sample_rate  # because we have complex sampled data. 

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate. '
    width_ratio = transition_width/nyq_rate

    taps = firwin(num_taps, 2*cutoff_hz/nyq_rate, window=window_type)

    return taps
    