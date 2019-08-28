import sys
import os

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from utils.experiment_options.experimentoptions import ExperimentOptions

STD_RF_RX_RATE = 5.0e6
RX_RATE_45KM = 10.0e3/3
RX_RATE_15KM = 10.0e3

SEQUENCE_7P = [0, 9, 12, 20, 22, 26, 27]
TAU_SPACING_7P = 2400 #us

SEQUENCE_8P = [0, 14, 22, 24, 27, 31, 42, 43]
TAU_SPACING_8P = 1500 #us

STD_8P_LAG_TABLE = [[ 0, 0],
                    [42,43],
                    [22,24],
                    [24,27],
                    [27,31],
                    [22,27],
                    [24,31],
                    [14,22],
                    [22,31],
                    [14,24],
                    [31,42],
                    [31,43],
                    [14,27],
                    [ 0,14],
                    [27,42],
                    [27,43],
                    [14,31],
                    [24,42],
                    [24,43],
                    [22,42],
                    [22,43],
                    [ 0,22],
                    [ 0,24],
                    [43,43]]

PULSE_LEN_45KM = 300 #us
PULSE_LEN_15KM = 100 #us

STD_16_BEAM_ANGLE = [-26.25, -22.75, -19.25, -15.75, -12.25, -8.75,
            -5.25, -1.75, 1.75, 5.25, 8.75, 12.25, 15.75, 19.25, 22.75,
            26.25]


STD_NUM_RANGES = 75
STD_FIRST_RANGE = 180  # km

STD_16_FORWARD_BEAM_ORDER = [0, 1, 2 , 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
STD_16_REVERSE_BEAM_ORDER = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

# Scanning directions here for now.
opts = ExperimentOptions()
IS_FORWARD_RADAR = IS_REVERSE_RADAR = False
if opts.site_id in ["sas", "rkn", "inv"]:
    IS_FORWARD_RADAR = True

if opts.site_id in ["cly", "pgr"]:
    IS_REVERSE_RADAR = True