import sys
import os
import numpy as np

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from utils.options.experimentoptions import ExperimentOptions
opts = ExperimentOptions()

# TODO: We should protect these values from changing, I noticed during testing that I used a
# TODO: call to reverse() on one and it affected the rest of the testing afterwards

STD_RF_RX_RATE = 5.0e6
RX_RATE_45KM = 10.0e3 / 3
RX_RATE_15KM = 10.0e3

SEQUENCE_7P = [0, 9, 12, 20, 22, 26, 27]
TAU_SPACING_7P = 2400  # us
INTT_7P = 3700

SEQUENCE_8P = [0, 14, 22, 24, 27, 31, 42, 43]
TAU_SPACING_8P = 1500  # us
INTT_8P = 3700

STD_8P_LAG_TABLE = [[0, 0],
                    [42, 43],
                    [22, 24],
                    [24, 27],
                    [27, 31],
                    [22, 27],
                    [24, 31],
                    [14, 22],
                    [22, 31],
                    [14, 24],
                    [31, 42],
                    [31, 43],
                    [14, 27],
                    [0, 14],
                    [27, 42],
                    [27, 43],
                    [14, 31],
                    [24, 42],
                    [24, 43],
                    [22, 42],
                    [22, 43],
                    [0, 22],
                    [0, 24],
                    [43, 43]]

PULSE_LEN_45KM = 300  # us
PULSE_LEN_15KM = 100  # us

STD_16_BEAM_ANGLE = [(float(opts.beam_sep) * (beam_dir - 15/2)) for beam_dir in range(0, 16)]

STD_NUM_RANGES = 75
POLARDARN_NUM_RANGES = 75
STD_FIRST_RANGE = 180  # km

STD_16_FORWARD_BEAM_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
STD_16_REVERSE_BEAM_ORDER = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

# Scanning directions here for now.
IS_FORWARD_RADAR = IS_REVERSE_RADAR = False
if opts.site_id in ["sas", "rkn", "inv"]:
    IS_FORWARD_RADAR = True

if opts.site_id in ["cly", "pgr"]:
    IS_REVERSE_RADAR = True

# set common mode operating frequencies with a slight offset.
if opts.site_id == "sas":
    COMMON_MODE_FREQ_1 = 10500
    COMMON_MODE_FREQ_2 = 13000
elif opts.site_id == "pgr":
    COMMON_MODE_FREQ_1 = 10600
    COMMON_MODE_FREQ_2 = 13100
elif opts.site_id == "rkn":
    COMMON_MODE_FREQ_1 = 10900
    COMMON_MODE_FREQ_2 = 12300
elif opts.site_id == "inv":
    COMMON_MODE_FREQ_1 = 10800
    COMMON_MODE_FREQ_2 = 12200
elif opts.site_id == "cly":
    COMMON_MODE_FREQ_1 = 10700
    COMMON_MODE_FREQ_2 = 12500
else:
    COMMON_MODE_FREQ_1 = 10400
    COMMON_MODE_FREQ_2 = 13200


def easy_scanbound(intt, beams):
    """
    Create integration time boundaries for the scan at the exact
    integration time (intt) boundaries. For new experiments, you
    may wish to ensure that your intt * len(beams) approaches a
    minute mark to reduce delay in waiting for the next scanbound.
    """
    return [i * (intt * 1e-3) for i in range(len(beams))]


# set sounding frequencies
if opts.site_id == "sas":
    SOUNDING_FREQS = [9690, 10500, 11000, 11700, 12400, 12900, 13150]
elif opts.site_id == "pgr":
    SOUNDING_FREQS = [9600, 10590, 11050, 11750, 13090, 12850, 12400]
elif opts.site_id == "rkn":
    SOUNDING_FREQS = [11100, 9600, 10500, 12400, 11800, 13090, 12825]
elif opts.site_id == "inv":
    SOUNDING_FREQS = [11150, 9690, 12400, 10590, 11850, 12800, 13100]
elif opts.site_id == "cly":
    SOUNDING_FREQS = [11900, 12400, 11100, 10400, 9600, 12800, 13050]
else:
    SOUNDING_FREQS = [10600, 11250, 11950, 13150]


def easy_widebeam(frequency_khz, tx_antennas, antenna_spacing_m):
    """
    Returns phases in degrees for each antenna in the main array that will generate a wide beam pattern
    that illuminates the full FOV. Only 8 or 16 antennas at common frequencies are supported.
    """
    if antenna_spacing_m != 15.24:
        raise ValueError("Antenna spacing must be 15.24m. Given value: {}".format(antenna_spacing_m))

    cached_values_16_antennas = {
        10400: [0., 33.21168501, 63.39856497, 133.51815213, 232.59694556, 287.65482653, 299.43588532, 313.30394893,
                313.30394893, 299.43588532, 287.65482653, 232.59694556, 133.51815213, 63.39856497, 33.21168501, 0.],
        10500: [0., 33.22157987, 63.44769218, 134.09072554, 232.41818196, 288.18043116, 299.96678003, 312.81034918,
                312.81034918, 299.96678003, 288.18043116, 232.41818196, 134.09072554, 63.44769218, 33.22157987, 0.],
        10600: [0., 33.49341546, 63.918406, 135.76673356, 232.41342064, 288.68373728, 299.8089564, 312.19755493,
                312.19755493, 299.8089564, 288.68373728, 232.41342064, 135.76673356, 63.918406, 33.49341546, 0.],
        10700: [0., 33.42706054, 63.94880958, 136.78441366, 232.43324622, 288.91978353, 299.57226291, 311.74840496,
                311.74840496, 299.57226291, 288.91978353, 232.43324622, 136.78441366, 63.94880958, 33.42706054, 0.],
        10800: [0., 33.13909903, 63.56879316, 137.23017826, 232.17488475, 289.01436937, 299.53525025, 311.23785241,
                311.23785241, 299.53525025, 289.01436937, 232.17488475, 137.23017826, 63.56879316, 33.13909903, 0.],
        10900: [0., 33.15305158, 63.55105706, 137.93590292, 232.13550152, 289.46328775, 299.78227805, 310.57614029,
                310.57614029, 299.78227805, 289.46328775, 232.13550152, 137.93590292, 63.55105706, 33.15305158, 0.],
        12200: [0., 70.91038811, 122.60927618, 214.92179098, 276.38784179, 325.25390655, 351.3873793, 316.5693829,
                316.5693829, 351.3873793, 325.25390655, 276.38784179, 214.92179098, 122.60927618, 70.91038811, 0.],
        12300: [0., 71.78224973, 124.29124213, 215.26781585, 277.84490172, 326.57004062, 353.22972278, 318.83181539,
                318.83181539, 353.22972278, 326.57004062, 277.84490172, 215.26781585, 124.29124213, 71.78224973, 0.],
        12500: [0., 75.1870308, 128.12468688, 216.50545923, 281.26273571, 334.23044519, 357.70997722, 326.41420518,
                326.41420518, 357.70997722, 334.23044519, 281.26273571, 216.50545923, 128.12468688, 75.1870308, 0.],
        13000: [0., 65.30441048, 122.04513377, 208.77532736, 282.14858123, 329.88094473, 368.67442895, 324.92709286,
                324.92709286, 368.67442895, 329.88094473, 282.14858123, 208.77532736, 122.04513377, 65.30441048, 0.],
        13100: [0., 75.41723909, 133.59413156, 216.03815626, 287.94258174, 343.50035796, 369.91299149, 337.96682569,
                337.96682569, 369.91299149, 343.50035796, 287.94258174, 216.03815626, 133.59413156, 75.41723909, 0.],
        13200: [0., 67.98474247, 126.21855408, 209.5839628, 285.48610109, 333.17276884, 370.37654775, 329.43903017,
                329.43903017, 370.37654775, 333.17276884, 285.48610109, 209.5839628, 126.21855408, 67.98474247, 0.]
    }
    cached_values_8_antennas = {
        10400: [0., 25.65596691, 78.37293679, 139.64736262, 139.64736262, 78.37293679, 25.65596691, 0.],
        10500: [0., 25.08958919, 77.59100768, 140.85808655, 140.85808655, 77.59100768, 25.08958919, 0.],
        10600: [0., 24.57335302, 76.75481191, 141.98499171, 141.98499171, 76.75481191, 24.57335302, 0.],
        10700: [0., 23.8098711,  75.90392693, 143.01444351, 143.01444351, 75.90392693, 23.8098711,  0.],
        10800: [0., 22.11931133, 73.23562257, 143.47732068, 143.47732068, 73.23562257, 22.11931133, 0.],
        10900: [0., 22.85211015, 72.76130323, 144.37536937, 144.37536937, 72.76130323, 22.85211015, 0.],
        12200: [0., 24.12132192, 67.43277427, 160.59421469, 160.59421469, 67.43277427, 24.12132192, 0.],
        12300: [0., 25.79888664, 68.32548572, 162.24856417, 162.24856417, 68.32548572, 25.79888664, 0.],
        12500: [0., 29.73310292, 70.83940609, 166.04550735, 166.04550735, 70.83940609, 29.73310292, 0.],
        13000: [0., 41.4313578,  82.16477044, 175.25809179, 175.25809179, 82.16477044, 41.4313578,  0.],
        13100: [0., 43.20693263, 84.14234248, 175.38631445, 175.38631445, 84.14234248, 43.20693263, 0.],
        13200: [0., 43.42908842, 84.21675093, 174.68458927, 174.68458927, 84.21675093, 43.42908842, 0.]
    }
    num_antennas = opts.main_antenna_count
    phases = np.zeros(num_antennas, dtype=np.complex64)
    if len(tx_antennas) == 16:
        if frequency_khz in cached_values_16_antennas.keys():
            phases[tx_antennas] = np.exp(1j * np.pi/180. * np.array(cached_values_16_antennas[frequency_khz]))
            return phases.reshape(1, num_antennas)
    elif len(tx_antennas) == 8:
        if frequency_khz in cached_values_8_antennas.keys():
            phases[tx_antennas] = np.exp(1j * np.pi/180. * np.array(cached_values_8_antennas[frequency_khz]))
            return phases.reshape(1, num_antennas)
    # If you get this far, the number of antennas or frequency is not supported for this function.
    raise ValueError("Invalid parameters for easy_widebeam(): tx_antennas: {}, frequency_khz: {}, "
                     "main_antenna_count: {}"
                     "".format(tx_antennas, frequency_khz, num_antennas))
