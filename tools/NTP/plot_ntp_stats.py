#!/usr/bin/python3

"""
    plot_ntp_stats process
    ~~~~~~~~~~~~~~~~~~~~~

    plot_ntp_stats has utility scripts to analyze and plot the statistics logged by NTP
    :copyright: 2020 SuperDARN Canada
    :author: Kevin Krieger

    Some references used:

    https://deltafabri.wordpress.com/2014/11/12/how-to-build-a-portable-stratum-1-ntp-server/
    https://www.phidgets.com/docs/Allan_Deviation_Primer
    http://www.ko4bb.com/getsimple/index.php?id=timing-faq-2-clock-stability-analysis-allan-deviation
    https://www.eecis.udel.edu/~mills/ntp/html/drivers/driver22.html
    https://www.eecis.udel.edu/~mills/ntp/html/pps.html
    https://www.eecis.udel.edu/~mills/ntp/html/kern.html
    https://www.eecis.udel.edu/~mills/ntp/html/kernpps.html
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse as ap
import allantools

POSSIBLE_FILE_TYPES = frozenset(['loopstats', 'clockstats', 'peerstats', 'rawstats', 'sysstats'])

# Loopstats present the NTP loop filter statistics.
# Each update of the local clock outputs a line containing the following:
LOOPSTATS_KEYS = [
    "modified_julian_day",    # MJD = Julian Day - 2400000.5 = Days since May 23, 1968, Int
    "seconds_past_midnight",  # How many seconds have elapsed since UTC midnight? Float
    "offset_seconds",         # Offset from real time, Float
    "freq_offset_ppm",        # Frequency offset in parts-per-million from true frequency, Float
    "polling_interval",       # Time constant of clock discipline algorithm in seconds. Int, power of 2
]

# Peerstats present the NTP peer statistics. Includes all stats records of all peers of the NTP server.
# Each valid update outputs a line containing the following:
PEERSTATS_KEYS = [
    "modified_julian_day",    # MJD = Julian Day - 2400000.5 = Days since May 23, 1968, Int
    "seconds_past_midnight",  # How many seconds have elapsed since UTC midnight? Float
    "peer_ip_address",        # The peer address in dotted-quad notation (example: 127.127.4.1)
    "status_field",           # The status field of the peer in hexadecimal. See APP A of RFC 1305
    "offset_seconds",         # The offset of the peer's clock from true time in seconds, Float
    "delay_seconds",          # The delay of the peer in seconds over the network, Float
    "dispersion_seconds",     # The dispersion of the peer in seconds, Float
]

# Clockstats present the NTP clock driver statistics.
# Each update from a clock driver outputs a line containing the following:
CLOCKSTATS_KEYS = [
    "modified_julian_day",    # MJD = Julian Day - 2400000.5 = Days since May 23, 1968, Int
    "seconds_past_midnight",  # How many seconds have elapsed since UTC midnight? Float
    "peer_ip_address",        # The peer address in dotted-quad notation (example: 127.127.4.1)
    "last_timecode",          # The last timecode received from the clock in ASCII format, Float
]

# Rawstats present the NTP raw-timestamp statistics. Includes all stats records of all peers of the NTP server.
# **NOTE** The timestamps entries are before any processing is done on the packet.
# Each NTP message received from a peer or clock driver outputs a line containing the following:
RAWSTATS_KEYS = [
    "modified_julian_day",    # MJD = Julian Day - 2400000.5 = Days since May 23, 1968, Int
    "seconds_past_midnight",  # How many seconds have elapsed since UTC midnight? Float
    "peer_ip_address",        # The peer address in dotted-quad notation (example: 127.127.4.1)
    "originate",              # The originate timestamp in seconds, Float
    "receive",                # The receive timestamp in seconds, Float
    "transmit",               # The transmit timestamp in seconds, Float
    "final",                  # The final timestamp in seconds, Float
]

# Sysstats present the NTP system statistics.
# Each hour the NTP system outputs a line containing the following:
SYSSTATS_KEYS = [
    "modified_julian_day",    # MJD = Julian Day - 2400000.5 = Days since May 23, 1968, Int
    "seconds_past_midnight",  # How many seconds have elapsed since UTC midnight? Float
    "time_since_restart",     # The number of hours elapsed since system last rebooted, Int
    "packets_received",       # The number of packets received, Int
    "packets_processed",      # The number of packets received in response to previous packets sent, Int
    "current_version",        # The number of packets received that matched the current version of NTP, Int
    "previous_version",       # The number of packets received that matched the previous version of NTP, Int
    "bad_version",            # The number of packets received that matched neither current nor prev version of NTP, Int
    "access_denied",          # The number of packets denied access for any reason, Int
    "bad_length",             # The number of packets with an invalid length, format or port number, Int
    "bad_authentication",     # The number of packets not verified as authentic, Int
    "rate_exceeded",          # The number of packets discarded due to rate limitation, Int
]

#PEER_STATUS_WORD = {
#    "status",   b'0000000000011111'     #
#    "select",   b'0000000011100000'     #
#    "count",    b'0000111100000000'     #
#    "code",     b'1111000000000000'     #
#}


# class NTPUtils(object):
# TODO: place static methods into an NTPUtils class

if __name__ == "__main__":
    # TODO: Cleanup arguments to allow a list of plot types, or to plot all from available files
    parser = ap.ArgumentParser(description='Analyze and plot NTP statistics')
    parser.add_argument('--allan-deviation', help='Enable plotting Allan deviation',
                        action='store_true')
    parser.add_argument('--offset', help='Plot the loopstats offset value (s)',
                        action='store_true')
    parser.add_argument('--freq-offset', help='Plot the loopstats frequency offset value (ppm)',
                        action='store_true')
    parser.add_argument('--peer-offset', help='Plot the peerstats estimated offset from true time (s)',
                        action='store_true')
    parser.add_argument('--peer-delay', help='Plot the peerstats packet delay over the network (s)',
                        action='store_true')
    parser.add_argument('--peer-dispersion', help='Plot the peerstats estimated dispersion (s)',
                        action='store_true')
    parser.add_argument('--ntp-config', help='Where is the NTP config file located?',
                        action='store', default="/etc/ntp.conf")
    parser.add_argument('input_file', help='The input statistics file')
    args = parser.parse_args()

    #ntpu = NTPUtils

    print("Analyzing {inputfile}".format(inputfile=args.input_file))

    # Get the ntp configuration file and grab specific options from it:
    # location of the statistics files
    # the names of every server
    # the fudge values for every server
    # Whether or not the PPS clockstats is being written every second
    try:
        ntp_config_options = None
        with open(args.ntp_config) as f:
            ntp_config_options = f.readlines()
    except OSError as e:
        print("WARNING: Could not open ntp config file: {}".format(e))
        print("You may not have accurate plots.")

    # TODO: Ability to plot using all files available in the ntpstats directory
    if not ntp_config_options:
        print("No ntp config file options loaded")
    else:
        ntp_statsdir = [line for line in ntp_config_options if "statsdir" in line]
        ntp_statsdir = ntp_statsdir[0].split()[1]
        ntp_servers = [line for line in ntp_config_options if "server" in line]
        ntp_server_addresses = []
        for server in ntp_servers:
            ntp_server_addresses.append(server.split()[1])
        ntp_fudges = [line for line in ntp_config_options if "fudge" in line]
        pps_clockstats_rate = None
        for fudge_line in ntp_fudges:
            if "127.127.22.0" in fudge_line:
                pps_fudge = fudge_line.split()
                if "flag4" in pps_fudge:
                    flag4_value_index = pps_fudge.index("flag4") + 1
                    if "1" in pps_fudge[flag4_value_index]:
                        # We know we have 1 second sampling for pps clock driver
                        pps_clockstats_rate = 1  # Hz
                break

        print("ntp stats dir: {}".format(ntp_statsdir))
        print("ntp servers: {}".format(ntp_server_addresses))
        print("ntp fudges: {}".format(ntp_fudges))
        print("PPS Clockstats rate: {}".format(pps_clockstats_rate))

    # Check that we have an appropriate file type
    proper_file_type = None
    for filetype in POSSIBLE_FILE_TYPES:
        if filetype in args.input_file:
            proper_file_type = filetype
            print("Input file type: {}".format(filetype))
    if not proper_file_type:
        raise ValueError("Invalid file type. Expected one of {}".format(POSSIBLE_FILE_TYPES))

    if args.allan_deviation:
        if proper_file_type != 'clockstats':
            raise ValueError("Invalid file type for producing AllanDeviation. Expected clockstats")

        print("Plotting Allan deviation")
        clock_times = np.loadtxt(args.input_file, usecols=3)
        rate = 1   # 1 Hz, if NTP is set up with PPS flag4 = 1
        at_dataset = allantools.Dataset(data=clock_times, rate=rate, data_type="phase", taus="octave")

        # Other possible computations are adev, mdev, tdev, hdev, ohdev, totdev,
        # mtotdev, ttotdev, htotdev, theo1, mtie, tierms, gradev
        at_dataset.compute("oadev")
        at_plotter = allantools.Plot()
        at_plotter.plot(at_dataset)
        at_plotter.show()

    # TODO: Code reuse for most of the following:
    if args.offset:
        if proper_file_type != 'loopstats':
            raise ValueError("Invalid file type for producing offset plot. Expected loopstats")

        print("Plotting loopstats offset")
        offsets = np.loadtxt(args.input_file, usecols=2)
        fig, ax = plt.subplots()
        t = np.arange(len(offsets))  # TODO: Plot this using actual date-time values from loopstats
        ax.plot(t*16, offsets)
        ax.set(xlabel='time (s)', ylabel='offset (s)', title='Loopstats offset')
        ax.grid()
        plt.show()

    if args.freq_offset:
        if proper_file_type != 'loopstats':
            raise ValueError("Invalid file type for producing freq offset plot. Expected loopstats")

        print("Plotting loopstats frequency offset")
        freqoffsets = np.loadtxt(args.input_file, usecols=3)
        fig, ax = plt.subplots()
        t = np.arange(len(freqoffsets))  # TODO: Plot this using actual date-time values from loopstats
        ax.plot(t * 16, freqoffsets)
        ax.set(xlabel='time (s)', ylabel='freq offset (ppm)', title='Loopstats freq offset (ppm)')
        ax.grid()
        plt.show()

    if args.peer_offset:
        if proper_file_type != 'peerstats':
            raise ValueError("Invalid file type for producing peer offset plot. Expected peerstats")

        print("Plotting peer offset")
        peeroffsets = np.loadtxt(args.input_file, usecols=4)
        fig, ax = plt.subplots()
        t = np.arange(len(peeroffsets))  # TODO: Plot this using actual date-time values from peerstats
        ax.plot(t * 16, peeroffsets)
        ax.set(xlabel='time (s)', ylabel='offset (s)', title='Peer offset')
        ax.grid()
        plt.show()

    if args.peer_delay:
        if proper_file_type != 'peerstats':
            raise ValueError("Invalid file type for producing peer delay plot. Expected peerstats")

        print("Plotting peer network delay")
        peerdelay = np.loadtxt(args.input_file, usecols=5)
        fig, ax = plt.subplots()
        t = np.arange(len(peerdelay))  # TODO: Plot this using actual date-time values from peerstats
        ax.plot(t * 16, peerdelay)
        ax.set(xlabel='time (s)', ylabel='Delay (s)', title='Peer delay')
        ax.grid()
        plt.show()

    if args.peer_dispersion:
        if proper_file_type != 'peerstats':
            raise ValueError("Invalid file type for producing peer dispersion plot. Expected peerstats")

        print("Plotting peer dispersion")
        peerdispersions = np.loadtxt(args.input_file, usecols=6)
        fig, ax = plt.subplots()
        t = np.arange(len(peerdispersions))  # TODO: Plot this using actual date-time values from peerstats
        ax.plot(t * 16, peerdispersions)
        ax.set(xlabel='time (s)', ylabel='Dispersion (s)', title='Peer dispersion')
        ax.grid()
        plt.show()
