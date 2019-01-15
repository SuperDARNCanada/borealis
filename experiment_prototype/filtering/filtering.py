
import sys
import math
from scipy import signal


BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from utils.experiment_options.experimentoptions import ExperimentOptions
from experiment_prototype.experiment_exception import ExperimentException

class Filter(object):

    def __init__(self, stage, input_rate, dm_rate, filter_taps):
        """

        :param stage:
        :param input_rate:
        :param dm_rate:
        :param filter_taps:
        """
        self.stage = stage
        self.input_rate = input_rate
        if not isinstance(dm_rate, int):
            raise ExperimentException('Decimation Rate is not an Integer')
        self.output_rate = input_rate/dm_rate
        self.dm_rate = dm_rate
        self.filter_taps = filter_taps


class FilteringScheme(object):
    """
    Class for DSP filtering scheme.
    """

    options = ExperimentOptions()

    def __init__(self, filters=None):
        self.input_rate = []
        self.dm_rates = []
        self.output_rates = []
        self.filter_gains = []
        if filters is None:
            # create the default filters, using remez.


        self.filters = []


    "first_stage_sample_rate" : "1.0e6",
    "second_stage_sample_rate" : "100.0e3",
    "first_stage_filter_cutoff" : "1.0e6",
    "first_stage_filter_transition" : "500.0e3",
    "second_stage_filter_cutoff" : "100.0e3",
    "second_stage_filter_transition" : "50.0e3",
    "third_stage_filter_cutoff" : "3.333e3",
    "third_stage_filter_transition" : "0.833e3",
    "first_stage_scaling_factor" : "1.0",
    "second_stage_scaling_factor" : "1.0",
    "third_stage_scaling_factor" : "12.0",

def calculate_num_filter_taps(sampling_freq, trans_width, k):
    """
    Calculates the number of filter taps required for the filter, using the sampling rate,
    transition width desired, and a k value which impacts filter performance (3 typical).
    :param sampling_freq: sampling rate, Hz.
    :param trans_width: transition bandwidth between cutoff of passband and stopband (Hz)
    :param k: value increasing filter performance with increasing value.
    :return: num_taps, number of taps for the FIR filter.
    """
    num_taps = k * (sampling_freq /trans_width)
    return num_taps

def create_remez_filter(sampling_freq, cutoff_freq, trans_width):
    """
    Creates the default filter for the experiment, if the filter taps are not provided.
    :return:
    """

    num_taps = calculate_num_filter_taps(sampling_freq, trans_width, 3)

    lpass = signal.remez(num_taps, [0, cutoff_freq, cutoff_freq + trans_width,
                                    0.5 * sampling_freq], [1, 0], Hz=sampling_freq)

    return lpass

def determine_decimation_rates(rxrate, output_rx_rate):
    """
    Comes up with decimation rates for four stages, if not provided.
    :return:
    """
    overall_rate = rxrate /output_rx_rate
    error = overall_rate - int(overall_rate)
    total_dm_rate = int(overall_rate)
    return total_dm_rate
