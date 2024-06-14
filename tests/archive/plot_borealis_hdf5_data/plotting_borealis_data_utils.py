#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import math


def reshape_bfiq_data(record_dict):
    """
    Reshape the flattened data from an bfiq file.

    :param record_dict: dict of the hdf5 data of a given record of bfiq, ie deepdish.io.load(filename)[record_name]
    """
    # data dimensions are num_antenna_arrays, num_sequences, num_beams, num_samps
    number_of_beams = len(record_dict["beam_azms"])
    number_of_arrays = len(record_dict["antenna_arrays_order"])
    flat_data = np.array(record_dict["data"])
    # reshape to 2 (main, intf) x nave x number_of_beams x number_of_samples
    bf_iq_data = np.reshape(
        flat_data,
        (
            number_of_arrays,
            record_dict["num_sequences"],
            number_of_beams,
            record_dict["num_samps"],
        ),
    )
    record_dict["data"] = bf_iq_data
    return record_dict


def reshape_antennas_iq(record_dict):
    """
    Reshape the flattened data from an antennas_iq file.

    :param record_dict: dict of the hdf5 data of a given record of antennas_iq, ie deepdish.io.load(filename)[record_name]
    """
    # data dimensions are num_antennas, num_sequences, num_samps
    number_of_antennas = len(record_dict["antenna_arrays_order"])

    flat_data = np.array(record_dict["data"])
    # reshape to number of antennas (M0..... I3) x nave x number_of_samples
    antennas_iq_data = np.reshape(
        flat_data,
        (number_of_antennas, record_dict["num_sequences"], record_dict["num_samps"]),
    )
    record_dict["data"] = antennas_iq_data
    return record_dict


def reshape_rawrf_data(record_dict):
    """
    Reshape the flattened data from an rawrf file.

    :param record_dict: dict of the hdf5 data of a given record of rawrf, ie deepdish.io.load(filename)[record_name]
    """

    # data dimensions are num_sequences, num_antennas, num_samps

    number_of_antennas = (
        record_dict["main_antenna_count"] + record_dict["intf_antenna_count"]
    )
    flat_data = np.array(record_dict["data"])
    # reshape to nave x number of antennas (M0..... I3) x number_of_samples

    raw_rf_data_data = np.reshape(
        flat_data,
        (record_dict["num_sequences"], number_of_antennas, record_dict["num_samps"]),
    )
    record_dict["data"] = raw_rf_data_data
    return record_dict


def reshape_txdata(record_dict):
    """
    Reshape the flattened data from an txdata file.

    :param record_dict: dict of the hdf5 data of a given record of txdata, ie deepdish.io.load(filename)[record_name]
    """
    # data dimensions are num_sequences, num_antennas, num_samps
    try:
        record_dict["tx_rate"] = record_dict["tx_rate"][0]
        record_dict["dm_rate"] = record_dict["dm_rate"][0]
    except IndexError:
        pass
    record_dict["rx_sample_rate"] = int(record_dict["tx_rate"] / record_dict["dm_rate"])
    print("Decimation rate error: {}".format(record_dict["dm_rate_error"]))
    print(record_dict["rx_sample_rate"])
    record_dict["data_descriptors"] = ["num_sequences", "num_antennas", "num_samps"]
    record_dict["data"] = record_dict["tx_samples"]  # tx['decimated_tx_samples']
    record_dict["antennas_present"] = record_dict["decimated_tx_antennas"][0]
    record_dict["dm_start_sample"] = 0

    return record_dict


def plot_bf_iq_data(record_dict, record_info_string, beam=0, sequence=0):
    """
    :param record_dict: dict of the hdf5 data of a given record of bfiq, ie deepdish.io.load(filename)[record_name]
    :param record_info_string: a string indicating the type of data being plotted (to be used on the plot legend). Should
     be bf_iq type, but there might be multiple slices or you may wish to otherwise identify
    :param beam: The beam number, indexed from 0. Assumed only one beam and plotting the first.
    """

    record_dict = reshape_bfiq_data(record_dict)
    # new data dimensions are num_antenna_arrays, num_sequences, num_beams, num_samps

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    print("Sequence number: {}".format(sequence))
    print("Beam number: {}".format(beam))
    for index in range(0, record_dict["data"].shape[0]):
        antenna_array = record_dict["antenna_arrays_order"][index]

        if antenna_array == "main":
            ax1.set_title(
                "Main Array {} sequence {}".format(record_info_string, sequence)
            )
            ax1.plot(
                np.arange(record_dict["num_samps"]),
                record_dict["data"][index, sequence, beam, :].real,
                label="Real {}".format(antenna_array),
            )
            ax1.plot(
                np.arange(record_dict["num_samps"]),
                record_dict["data"][index, sequence, beam, :].imag,
                label="Imag {}".format(antenna_array),
            )
            ax1.legend()
        else:
            ax2.set_title(
                "Intf Array {} sequence {}".format(record_info_string, sequence)
            )
            ax2.plot(
                np.arange(record_dict["num_samps"]),
                record_dict["data"][index, sequence, beam, :].real,
                label="Real {}".format(antenna_array),
            )
            ax2.plot(
                np.arange(record_dict["num_samps"]),
                record_dict["data"][index, sequence, beam, :].imag,
                label="Imag {}".format(antenna_array),
            )
            ax2.legend()
    plt.show()


def plot_antennas_iq_data(
    record_dict, record_info_string, sequence=0, real_only=True, antenna_indices=None
):
    """
    :param record_dict: dict of the hdf5 data of a given record of antennas_iq, ie deepdish.io.load(filename)[record_name]
    :param record_info_string: a string indicating the type of data being plotted (to be used on the plot legend). Should
     be antennas_iq type, but there might be multiple slices.
    """

    record_dict = reshape_antennas_iq(record_dict)
    # new data dimensions are num_antennas, num_sequences, num_samps
    antennas_present = [
        int(i.split("_")[-1]) for i in record_dict["antenna_arrays_order"]
    ]

    if antenna_indices is None:
        indices = range(0, record_dict["data"].shape[0])
    else:
        indices = antenna_indices

    print("Sequence number: {}".format(sequence))
    print(
        "Antennas: {}".format([record_dict["antenna_arrays_order"][i] for i in indices])
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    for index in indices:
        antenna = antennas_present[index]
        ax1.set_title("Main Antennas {}".format(record_info_string))
        ax2.set_title("Intf Antennas {}".format(record_info_string))
        if antenna < record_dict["main_antenna_count"]:
            ax1.plot(
                np.arange(record_dict["num_samps"]),
                record_dict["data"][index, sequence, :].real,
                label="Real {}".format(antenna),
            )
            if not real_only:
                ax1.plot(
                    np.arange(record_dict["num_samps"]),
                    record_dict["data"][index, sequence, :].imag,
                    label="Imag {}".format(antenna),
                )
            ax1.legend()
        else:
            ax2.plot(
                np.arange(record_dict["num_samps"]),
                record_dict["data"][index, sequence, :].real,
                label="Real {}".format(antenna),
            )
            if not real_only:
                ax2.plot(
                    np.arange(record_dict["num_samps"]),
                    record_dict["data"][index, sequence, :].imag,
                    label="Imag {}".format(antenna),
                )
            ax2.legend()
    plt.show()


def plot_output_raw_data(
    record_dict,
    record_info_string,
    sequence=0,
    real_only=True,
    start_sample=18000,
    end_sample=20000,
    antenna_indices=None,
):
    """
    :param record_dict: dict of the hdf5 data of a given record of rawrf, ie deepdish.io.load(filename)[record_name]
    :param record_info_string: a string indicating the type of data being plotted (to be used on the plot legend). Should
     be raw_rf_data type, but you may also want to distinguish what data this is.
    """

    record_dict = reshape_rawrf_data(record_dict)
    # new data dimensions are num_sequences, num_antennas, num_samps
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title("Main Antennas {}".format(record_info_string))
    ax2.set_title("Intf Antennas {}".format(record_info_string))

    if antenna_indices is None:
        indices = range(0, record_dict["data"].shape[1])
    else:
        indices = antenna_indices

    print("Sequence number: {}".format(sequence))
    print("Antennas: {}".format(indices))
    print("Sample number {} to {}".format(start_sample, end_sample))
    for antenna in indices:
        # if max(abs(record_dict['data'][index,antenna,18000:20000])) < 0.05:
        #    continue
        if antenna < record_dict["main_antenna_count"]:
            ax1.plot(
                np.arange(end_sample - start_sample),
                record_dict["data"][sequence, antenna, start_sample:end_sample].real,
                label="Real antenna {}".format(antenna),
            )
            if not real_only:
                ax1.plot(
                    np.arange(end_sample - start_sample),
                    record_dict["data"][
                        sequence, antenna, start_sample:end_sample
                    ].imag,
                    label="Imag {}".format(antenna),
                )
        else:
            ax2.plot(
                np.arange(end_sample - start_sample),
                record_dict["data"][sequence, antenna, start_sample:end_sample].real,
                label="Real antenna {}".format(antenna),
            )
            if not real_only:
                ax2.plot(
                    np.arange(end_sample - start_sample),
                    record_dict["data"][
                        sequence, antenna, start_sample:end_sample
                    ].imag,
                    label="Imag {}".format(antenna),
                )
    ax1.legend()
    ax2.legend()
    plt.show()


def plot_output_tx_data(
    record_dict, record_info_string, sequence=0, real_only=True, antenna_indices=None
):
    """
    :param record_dict: dict of the hdf5 data of a given record of txdata, ie deepdish.io.load(filename)[record_name]
    :param record_info_string): a string indicating the type of data being plotted (to be used on the plot legend). Should
     be raw_rf_data type, but you may want to otherwise identify the data.
    """

    record_dict = reshape_txdata(record_dict)
    # new data dimensions are num_sequences, num_antennas, num_samps

    if antenna_indices is None:
        indices = range(0, record_dict["data"].shape[1])
    else:
        indices = antenna_indices

    print("Sequence number: {}".format(sequence))
    print("Antennas: {}".format(indices))
    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    plt.title(record_info_string)
    for antenna in indices:
        # if max(abs(record_dict['data'][index,antenna,18000:20000])) < 0.05:
        #    continue
        ax1.plot(
            np.arange(record_dict["data"].shape[2]),
            record_dict["data"][sequence, antenna, :].real,
            label="Real antenna {}".format(antenna),
        )
        if not real_only:
            ax1.plot(
                np.arange(record_dict["data"].shape[2]),
                record_dict["data"][sequence, antenna, :].imag,
                label="Imag antenna {}".format(antenna),
            )
    ax1.legend()
    plt.show()


def fft_and_plot_bfiq_data(
    record_dict, record_info_string, beam=0, sequence=0, plot_width=None
):
    """
    :param plot_width: frequency bandwidth to plot fft (for higher resolution)
    """

    record_dict = reshape_bfiq_data(record_dict)
    # new data dimensions are num_antenna_arrays, num_sequences, num_beams, num_samps

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    print("Sequence number: {}".format(sequence))
    print("Beam number: {}".format(beam))
    for index in range(0, record_dict["data"].shape[0]):
        antenna_array = record_dict["antenna_arrays_order"][index]
        if not plot_width:
            fft_samps, xf = fft_to_plot(
                record_dict["data"][index, sequence, beam, :],
                record_dict["rx_sample_rate"],
            )
        else:
            fft_samps, xf = fft_to_plot(
                record_dict["data"][index, sequence, beam, :],
                record_dict["rx_sample_rate"],
                plot_width=plot_width,
            )
        len_samples = len(record_dict["data"][index, sequence, beam, :])
        if antenna_array == "main":
            ax1.set_title(
                "FFT Main Array {} sequence {}".format(record_info_string, sequence)
            )
            ax1.plot(
                xf,
                1.0 / len(fft_samps) * np.abs(fft_samps),
                label="{}".format(antenna_array),
            )
        else:
            ax2.set_title(
                "FFT Intf Array {} sequence {}".format(record_info_string, sequence)
            )
            ax2.plot(
                xf,
                1.0 / len(fft_samps) * np.abs(fft_samps),
                label="{}".format(antenna_array),
            )
    ax2.set_xlabel("Hz")
    return fft_samps, xf, fig


def fft_and_plot_antennas_iq(
    record_dict,
    record_info_string,
    sequence=0,
    real_only=True,
    antenna_indices=None,
    plot_width=None,
):
    """
    :param record_dict: dict of the hdf5 data of a given record of antennas_iq, ie deepdish.io.load(filename)[record_name]
    :param record_info_string: a string indicating the type of data being plotted (to be used on the plot legend). Should
     be antennas_iq type, but there might be multiple slices.
    :param plot_width: frequency bandwidth to plot fft (for higher resolution)
    """
    plot_individual = True
    plt.rcParams.update({"figure.max_open_warning": 0})

    record_dict = reshape_antennas_iq(record_dict)
    # new data dimensions are num_antennas, num_sequences, num_samps
    antennas_present = [
        int(i.split("_")[-1]) for i in record_dict["antenna_arrays_order"]
    ]

    if antenna_indices is None:
        indices = range(0, record_dict["data"].shape[0])
    else:
        indices = antenna_indices

    print("Sequence number: {}".format(sequence))
    print(
        "Antennas: {}".format([record_dict["antenna_arrays_order"][i] for i in indices])
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    for index in indices:
        antenna = antennas_present[index]
        plt.figure(0)
        ax1.set_title(
            "FFT Main Antennas {}".format(record_info_string) + ": All antennas"
        )
        ax2.set_title(
            "FFT Intf Antennas {}".format(record_info_string) + ": All antennas"
        )
        if not plot_width:
            fft_samps, xf = fft_to_plot(
                record_dict["data"][index, sequence, :], record_dict["rx_sample_rate"]
            )
        else:
            fft_samps, xf = fft_to_plot(
                record_dict["data"][index, sequence, :],
                record_dict["rx_sample_rate"],
                plot_width=plot_width,
            )
        len_samples = len(record_dict["data"][index, sequence, :])
        if antenna < record_dict["main_antenna_count"]:
            ax1.plot(
                xf, 1.0 / len_samples * np.abs(fft_samps), label="{}".format(antenna)
            )
        else:
            ax2.plot(
                xf, 1.0 / len_samples * np.abs(fft_samps), label="{}".format(antenna)
            )

        # Plot individual antenna on separate plot
        if plot_individual:
            plt.figure(index + 2, figsize=((6, 2)))
            plt.plot(
                xf, 1.0 / len_samples * np.abs(fft_samps), label="{}".format(antenna)
            )
            plt.title("Antenna " + str(index))
    ax2.set_xlabel("Hz")
    return fft_samps, xf, fig


def fft_and_plot_rawrf_data(
    record_dict,
    record_info_string,
    sequence=0,
    real_only=True,
    start_sample=18000,
    end_sample=20000,
    antenna_indices=None,
    plot_width=None,
    center=0,
):
    """
    :param plot_width: frequency bandwidth to plot fft (for higher resolution)
    """

    record_dict = reshape_rawrf_data(record_dict)
    # new data dimensions are num_sequences, num_antennas, num_samps
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title("Main Antennas {}".format(record_info_string))
    ax2.set_title("Intf Antennas {}".format(record_info_string))

    if antenna_indices is None:
        indices = range(0, record_dict["data"].shape[1])
    else:
        indices = antenna_indices

    print("Sequence number: {}".format(sequence))
    print("Antennas: {}".format(indices))
    print("Sample number {} to {}".format(start_sample, end_sample))
    print("TODO print center frequency once rawrf files have that included.")

    for antenna in indices:
        # if max(abs(record_dict['data'][index,antenna,18000:20000])) < 0.05:
        #    continue
        fft_samps, xf = fft_to_plot(
            record_dict["data"][sequence, antenna, start_sample:end_sample],
            record_dict["rx_sample_rate"],
            plot_width=plot_width,
            center=center,
        )
        len_samples = len(
            record_dict["data"][sequence, antenna, start_sample:end_sample]
        )
        if antenna < record_dict["main_antenna_count"]:
            ax1.plot(
                xf, 1.0 / len_samples * np.abs(fft_samps), label="{}".format(antenna)
            )
        else:
            ax2.plot(
                xf, 1.0 / len_samples * np.abs(fft_samps), label="{}".format(antenna)
            )
    ax1.legend()
    ax2.legend()
    ax2.set_xlabel("Hz")
    return fft_samps, xf, fig


def fft_and_plot_txdata(
    record_dict,
    record_info_string,
    sequence=0,
    real_only=True,
    antenna_indices=None,
    plot_width=None,
):

    record_dict = reshape_txdata(record_dict)
    # new data dimensions are num_sequences, num_antennas, num_samps

    if antenna_indices is None:
        indices = range(0, record_dict["data"].shape[1])
    else:
        indices = antenna_indices

    print("Sequence number: {}".format(sequence))
    print("Antennas: {}".format(indices))
    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    plt.title(record_info_string)
    for antenna in indices:
        # if max(abs(record_dict['data'][index,antenna,18000:20000])) < 0.05:
        #    continue
        if not plot_width:
            fft_samps, xf = fft_to_plot(
                record_dict["data"][sequence, antenna, :], record_dict["tx_rate"]
            )
        else:
            fft_samps, xf = fft_to_plot(
                record_dict["data"][sequence, antenna, :],
                record_dict["tx_rate"],
                plot_width=plot_width,
            )
        len_samples = len(record_dict["data"][sequence, antenna, :])
        ax1.plot(xf, 1.0 / len_samples * np.abs(fft_samps), label="{}".format(antenna))
    ax1.legend()
    ax1.set_xlabel("Hz")
    return fft_samps, xf, fig


def fft_to_plot(samples, rate, plot_width=None, center=0):
    fft_samps = fft(samples)
    T = 1.0 / float(rate)
    num_samps = len(samples)
    xf = np.linspace(-1.0 / (2.0 * T), 1.0 / (2.0 * T), num_samps)  # Hz

    fft_to_plot = np.empty([num_samps], dtype=np.complex64)
    halfway = int(math.ceil(float(num_samps) / 2))
    fft_to_plot = np.concatenate([fft_samps[halfway:], fft_samps[:halfway]])
    # xf = xf[halfway-200:halfway+200]
    if not plot_width:
        return fft_to_plot, xf
    else:
        first_sample = int((rate / 2 + center - plot_width / 2) * num_samps / rate)
        end_sample = int((rate / 2 + center + plot_width / 2) * num_samps / rate)
        print(first_sample, end_sample)
        return fft_to_plot[first_sample:end_sample], xf[first_sample:end_sample]


def find_fft_peaks(fft_samples, fft_x):
    """
    Find the peaks of the fft by looking for the max values.

    """
