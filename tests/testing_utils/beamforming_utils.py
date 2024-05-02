import math
import numpy as np
from borealis.sample_building.sample_building import get_phase_shift, shift_samples


def fft_and_plot(samples, rate):
    """
    Plot the fft of the samples, given the rate. Shows the plot.
    This provides a two-sided FFT.
    :param samples: time domain samples to provide the FFT of.
    :param rate: the rate that the samples were taken at.
    """

    fft_samps = fft(samples)
    T = 1.0 / float(rate)
    num_samps = len(samples)
    xf = np.linspace(-1.0 / (2.0 * T), 1.0 / (2.0 * T), num_samps)

    fig, smpplt = plt.subplots(1, 1)

    fft_to_plot = np.empty([num_samps], dtype=np.complex64)
    halfway = int(math.ceil(float(num_samps) / 2))
    fft_to_plot = np.concatenate([fft_samps[halfway:], fft_samps[:halfway]])
    # xf = xf[halfway-200:halfway+200]
    # fft_to_plot = fft_to_plot[halfway-200:halfway+200]
    smpplt.plot(xf, 1.0 / num_samps * np.abs(fft_to_plot))
    plt.show()
    # return fig


def align_tx_samples(tx_samples, offset, array_len):
    """
    To align tx samples to received samples, provide an offset value
    according to where the signal best correlated and provide the length so that
    zeros can be padded around the signal in order to line it up with the received samples.
    :param tx_samples: samples to offset
    :param offset: number of zeros to place before the samples
    :array_len: new length that you would like the samples to be
    :returns aligned_ant_samples: sample array containing tx_samples with  the offset number
    of zeros at the front and with the length array_len.
    """

    zeros_offset = np.array([0.0] * (offset))
    # print('Shapes of arrays:{}, {}'.format(zeros_offset.shape, ant_samples[0].shape))
    aligned_ant_samples = np.concatenate((zeros_offset, tx_samples))

    if array_len > len(aligned_ant_samples):
        zeros_extender = np.array([0.0] * (array_len - len(aligned_ant_samples)))
        # print(len(zeros_extender))
        aligned_ant_samples = np.concatenate((aligned_ant_samples, zeros_extender))
    else:
        aligned_ant_samples = aligned_ant_samples[:array_len]

    return aligned_ant_samples


def correlate_and_align_tx_samples(tx_samples, some_other_samples):
    """

    :param tx_samples: array of tx samples
    :param some_other_samples: an arry of other samples at same sampling rate as tx_samples
    """
    corr = np.correlate(tx_samples, some_other_samples, mode="full")
    max_correlated_index = np.argmax(np.abs(corr))
    # print('Max index {}'.format(max_correlated_index))
    correlated_offset = max_correlated_index - len(tx_samples) + 2
    # TODO: why plus 2? figured out based on plotting. symptom of the 'full' correlation?
    # print('Correlated offset = {}'.format(correlated_offset))
    aligned_ant_samples = align_tx_samples(
        tx_samples, correlated_offset, len(some_other_samples)
    )

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,sharex=True)
    # ax1.plot(np.arange(len(tx_samples)),np.abs(tx_samples))
    # ax2.plot(np.arange(len(aligned_ant_samples)),np.abs(aligned_ant_samples))
    # ax3.plot(np.arange(len(some_other_samples)),np.abs(some_other_samples))
    # ax4.plot(np.arange(len(corr)),np.abs(corr))
    # plt.show()
    return aligned_ant_samples, correlated_offset


def beamform(antennas_data, beamdirs, rxfreq, antenna_spacing):
    """
    :param antennas_data: numpy array of dimensions num_antennas x num_samps. All antennas are assumed to be
    from the same array and are assumed to be side by side with antenna spacing 15.24 m, pulse_shift = 0.0
    :param beamdirs: list of azimuthal beam directions in degrees off boresite
    :param rxfreq: frequency to beamform at.
    :param antenna_spacing: spacing in metres between antennas, used to get the phase shift that
    corresponds to an azimuthal direction.
    """

    beamformed_data = []
    for beam_direction in beamdirs:
        # print(beam_direction)
        antenna_phase_shifts = []
        for antenna in range(0, antennas_data.shape[0]):
            # phase_shift = get_phshift(beam_direction, rxfreq, antenna, 0.0, 16, 15.24)
            phase_shift = math.fmod(
                (
                    -1
                    * get_phase_shift(
                        beam_direction,
                        rxfreq,
                        antenna,
                        0.0,
                        antennas_data.shape[0],
                        antenna_spacing,
                    )
                ),
                2 * math.pi,
            )
            antenna_phase_shifts.append(phase_shift)
        phased_antenna_data = [
            shift_samples(antennas_data[i], antenna_phase_shifts[i], 1.0)
            for i in range(0, antennas_data.shape[0])
        ]
        phased_antenna_data = np.array(phased_antenna_data)
        one_beam_data = np.sum(phased_antenna_data, axis=0)
        beamformed_data.append(one_beam_data)
    beamformed_data = np.array(beamformed_data)

    return beamformed_data


def get_offsets(samples_a, samples_b):
    """
    Return the offset of samples b from samples a.
    :param samples_a: numpy array of complex samples
    :param samples_b: another numpy array of complex samples
    """
    samples_diff = samples_a * np.conj(samples_b)
    # Gives you an array with numbers that have the magnitude of |samples_a| * |samples_b| but
    # the angle of angle(samples_a) - angle(samples_b)
    phase_offsets = np.angle(samples_diff)
    phase_offsets = phase_offsets * 180.0 / math.pi
    return list(phase_offsets)


def find_pulse_indices(
    data, threshold
):  # TODO change to not normalized and change to only have one sample if many sequential samples pass (from the same pulse)
    """
    :param data: a numpy array of complex values to find the pulses within.
    :param threshold: a magnitude valude threshold (absolute value) that pulses will be defined as being greater than.
    """
    absolute_max = max(abs(data))
    normalized_data = abs(data) / absolute_max
    pulse_points = normalized_data > threshold
    pulse_indices = list(np.where(pulse_points == True)[0])
    return pulse_indices


def plot_antenna_data(array_of_data, list_of_antennas, record_filetype):
    """
    :param record_dict: a numpy array of data, dimensions num_antennas x num_samps
    :param list_of_antennas: list of antennas present, for legend
    :param record_filetype: a string indicating the type of data being plotted (to be used on the plot legend). Should
     be txdata type, but there might be multiple slices.
    """

    antennas_present = list_of_antennas
    for index in range(0, array_of_data.shape[0]):
        antenna = antennas_present[index]
        fig, ax1 = plt.subplots(1, 1, sharex=True)
        ax1.set_title("Samples {}".format(record_filetype))
        ax1.plot(
            np.arange(array_of_data.shape[1]),
            array_of_data[index, :].real,
            label="Real {}".format(antenna),
        )
        ax1.plot(
            np.arange(array_of_data.shape[1]),
            array_of_data[index, :].imag,
            label="Imag {}".format(antenna),
        )
        ax1.legend()
        plt.show()


def plot_all_bf_data(record_data):
    """
    :param record_data: dictionary with keys = filetypes, each value is a record dictionaries from that filetype,
     for the same record identifier. (eg. '1543525820193')
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    for filetype, record_dict in record_data.items():
        # normalized_real = record_dict['main_bf_data'][0].real / max(abs(record_dict['main_bf_data'][0]))
        # normalized_imag = record_dict['main_bf_data'][0].imag / max(abs(record_dict['main_bf_data'][0]))
        # ax1.plot(np.arange(record_dict['main_bf_data'].shape[1]), normalized_real, label='Normalized Real {}'.format(filetype))
        # ax1.plot(np.arange(record_dict['main_bf_data'].shape[1]), normalized_imag, label="Normalized Imag {}".format(filetype))
        # ax1.legend()
        # normalized_real = record_dict['intf_bf_data'][0].real / max(abs(record_dict['intf_bf_data'][0]))
        # normalized_imag = record_dict['intf_bf_data'][0].imag / max(abs(record_dict['intf_bf_data'][0]))
        # ax2.plot(np.arange(record_dict['intf_bf_data'].shape[1]), normalized_real, label='INTF Normalized Real {}'.format(filetype))
        # ax2.plot(np.arange(record_dict['intf_bf_data'].shape[1]), normalized_imag, label="INTF Normalized Imag {}".format(filetype))
        # ax2.legend()
        ax1.plot(
            np.arange(record_dict["main_bf_data"].shape[1]),
            record_dict["main_bf_data"][0].real,
            label="Real {}".format(filetype),
        )
        ax1.plot(
            np.arange(record_dict["main_bf_data"].shape[1]),
            record_dict["main_bf_data"][0].imag,
            label="Imag {}".format(filetype),
        )
        ax2.plot(
            np.arange(record_dict["intf_bf_data"].shape[1]),
            record_dict["intf_bf_data"][0].real,
            label="INTF Real {}".format(filetype),
        )
        ax2.plot(
            np.arange(record_dict["intf_bf_data"].shape[1]),
            record_dict["intf_bf_data"][0].imag,
            label="INTF Imag {}".format(filetype),
        )
    ax1.legend()
    ax2.legend()
    plt.show()


def check_for_equal_samples_across_channels(
    record_dict, start_sample, end_sample, max_absolute_error, main_antenna_count
):
    """
    :param record_dict: has a key 'data' with all the data from all the channels to check
    """

    antennas = record_dict["antennas_present"]

    sequence_to_antenna_problem_dict = {}
    data_dimensions_list = record_dict["data_descriptors"]
    if all(data_dimensions_list == ["num_sequences", "num_antennas", "num_samps"]):
        data = record_dict["data"][:, 0:main_antenna_count, start_sample:end_sample]
        average_sequence_data = np.mean(data, axis=1)
        for sequence in range(0, data.shape[0]):
            antennas_out = []
            for antenna in range(0, data.shape[1]):
                equality = [
                    math.isclose(x, y, abs_tol=max_absolute_error)
                    for x, y in zip(
                        data[sequence, antenna, :], average_sequence_data[sequence, :]
                    )
                ]
                if not all(equality):
                    antennas_out.append(antenna)
            sequence_to_antenna_problem_dict[sequence] = antennas_out
    elif all(data_dimensions_list == ["num_antennas", "num_sequences", "num_samps"]):
        data = record_dict["data"][0:main_antenna_count, :, start_sample:end_sample]
        average_sequence_data = np.mean(data, axis=0)
        for sequence in range(0, data.shape[1]):
            antennas_out = []
            for antenna in range(0, data.shape[0]):
                equality = [
                    math.isclose(x, y, abs_tol=max_absolute_error)
                    for x, y in zip(
                        data[antenna, sequence, :], average_sequence_data[sequence, :]
                    )
                ]
                if not all(equality):
                    antennas_out.append(antenna)
            sequence_to_antenna_problem_dict[sequence] = antennas_out
    else:
        print(
            "Do not know how to compare data with dimensions = {}".format(
                data_dimensions_list
            )
        )
    print(
        "Sequence to Antenna Problem Dictionary: {}".format(
            sequence_to_antenna_problem_dict
        )
    )
    return sequence_to_antenna_problem_dict
