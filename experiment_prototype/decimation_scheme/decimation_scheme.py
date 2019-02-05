
import os
import sys
import math
from scipy import signal


BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from utils.experiment_options.experimentoptions import ExperimentOptions
from experiment_prototype.experiment_exception import ExperimentException
from functools import reduce

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
            raise ExperimentException('Decimation Rate is not an Integer')
        self.output_rate = input_rate/dm_rate
        self.dm_rate = dm_rate
        if not isinstance(filter_taps, list):
            errmsg = 'Filter taps {} must be a list in decimation stage {}'.format(filter_taps,
                                                                                   stage_num)
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
        options = ExperimentOptions()
        self.__num_stages = options.number_of_filtering_stages
        self.rxrate = rxrate
        self.output_sample_rate = output_sample_rate

        if stages is None:  # create the default filters, using remez.
            # Currently only creating default filters if sampling rate and output rate are set
            # up as per original design. TODO: make this more general.
            if rxrate != 5.0e6 or round(output_sample_rate, 0) != 3.333e3:
                errmsg = 'Default filters not defined for rxrate {} and output rate {}'.format(
                    rxrate, output_sample_rate)
                raise ExperimentException(errmsg)

            # set up defaults as per design.
            self.dm_rates = [5, 10, 30, 1]
            total_rates = [reduce((lambda a, b: a * b), decimation_list) for decimation_list in
                           [self.dm_rates[:x] for x in range(1, len(self.dm_rates) + 1)]]
            self.output_rates = [rxrate / x for x in total_rates]
            input_rates = [rxrate]
            input_rates.extend(self.output_rates[:-1])
            self.input_rates = input_rates
            self.filter_scaling_factors = [1.0, 1.0, 12.0, 1.0]

            # defaults for remez filters for first three stages per design.
            filter_transition_widths = [500.0e3, 50.0e3, 0.833e3]  # Hz
            filter_cutoffs = [1.0e6, 100.0e3, 3.333e3]  # Hz

            stages = []
            for stage_num in range(0, self.num_stages):
                dm_rate = self.dm_rates[stage_num]
                input_rate = self.input_rates[stage_num]
                filter_scaling_factor = self.filter_scaling_factors[stage_num]

                if stage_num in range(0, 3):
                    transition_width = filter_transition_widths[stage_num]
                    cutoff = filter_cutoffs[stage_num]
                    unity_gain_filter_taps = create_remez_filter(input_rate, cutoff,
                                                                 transition_width)
                else:  # append last stage - default is decimation = 1, filter = [1]
                    unity_gain_filter_taps = [1.0]

                filter_taps = [filter_scaling_factor * i for i in unity_gain_filter_taps]

                dec_stage = DecimationStage(stage_num, input_rate, dm_rate, filter_taps)
                stages.append(dec_stage)
            self.stages = stages
        else:
            # check that number of stages is correct
            if len(stages) != self.num_stages:
                errmsg = 'Decimation stages provided do not meet the required number of stages ' \
                         '{}'.format(self.num_stages)
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

            for stage_num in range(0, self.num_stages -1):
                if self.output_rates[stage_num] != self.input_rates[stage_num + 1]:
                    errmsg = 'Decimation stage {} output rate {} does not equal next stage {} ' \
                             'input rate {}'.format(stage_num, self.output_rates[stage_num],
                                                    stage_num + 1, self.input_rates[stage_num + 1])
                    raise ExperimentException(errmsg)

            if self.output_rates[-1] != self.output_sample_rate:
                errmsg = 'Last decimation stage {} does not have output rate {} equal to ' \
                         'requested output data rate {}'.format(self.num_stages - 1,
                                                                self.output_rates[-1],
                                                                self.output_sample_rate)
                raise ExperimentException(errmsg)

    def __repr__(self):
        repr_str = 'Decimation Scheme with {} stages:\n'.format(self.num_stages)
        for stage in self.stages:
            repr_str += '\nStage {}:'.format(stage.stage_num)
            repr_str += '\nInput Rate: {} Hz'.format(stage.input_rate)
            repr_str += '\nDecimation by: {}'.format(stage.dm_rate)
            repr_str += '\nOutput Rate: {} Hz'.format(stage.output_rate)
            repr_str += '\nNum taps: {}'.format(len(stage.filter_taps))
            #repr_str += '\nFilter Taps: {}\n'.format(stage.filter_taps)
        return repr_str

    @property
    def num_stages(self):
        return self.__num_stages

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

    lpass = signal.remez(num_taps, [0, cutoff_freq, cutoff_freq + trans_width,
                                    0.5 * sampling_freq], [1, 0], Hz=sampling_freq)

    return lpass
    