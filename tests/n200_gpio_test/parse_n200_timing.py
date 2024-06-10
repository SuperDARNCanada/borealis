#! /usr/bin/env python

# This script takes in a csv exported file from the Saleae Logic software
# and finds out if the timing passes or fails
import csv
import sys


class Switch(object):
    # Nabbed this class from http://code.activestate.com/recipes/410692/
    # This class provides the functionality we want. You only need to look at
    # this if you want to know how this works. It only needs to be defined
    # once, no need to muck around with its internals.
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:  # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False


# Columns in the csv file correspond to certain information.
# There is a timestamp, a tr bit and an rf bit.
time_index = 0
tr_index = 1
rf_index = 2

# The time delays between each of the various events can be programmed, so here we can change them
tr_pre_delay_us = 64.6  # Delay between TR high and RF pulse high
rf_pulse_time_us = 298.9  # How long is the RF pulse?
tr_post_delay_us = 57  # Delay between RF low and TR low
THRESHOLD_US = 1  # What error are we willing to live with?
between_ave_periods_s = (
    0.2  # What is the threshold between averaging periods to wait for?
)

# Items related to the pulse sequence
tau_us = 1500
pulse_sequence = [0, 14, 22, 24, 27, 31, 42, 43]  # This is the normal 8 pulse sequence
# pulse_sequence = [0,5,8,12,18,30,32,33]  # This is another 8 pulse sequence, a golomb sequence

# State machine
states = (
    "unknown",
    "between_sequences",
    "tr_pre_pulse",
    "pulse",
    "tr_post_pulse",
    "between_pulses",
)
state = states[0]  # Unknown

# Get the file and read into memory
test_file = None
try:
    test_file = csv.reader(open(sys.argv[1], "r"), delimiter=",")
except IOError as e:
    print(e)
    exit(1)

max_time_error_us = 0.0
timestamp = 0.0
pulse_num = 0
num_pulses = len(pulse_sequence)

previous_bits_value = 0
first_row = 1
for row_num, row in enumerate(test_file):
    if first_row:
        first_row = 0
        continue
    # There are 4 possible cases for the 2 bits, so let's just handle them all.
    bits_value = int(row[tr_index]) << tr_index - 1
    bits_value += int(row[rf_index]) << rf_index - 1
    previous_bits_value = bits_value
    timestamp_previous = timestamp
    timestamp = float(row[0])
    if __debug__:
        print("\n" + str(row_num) + " State: " + state)
        print("Pulse number: " + str(pulse_num))
        print("Bits value : " + str(bits_value))

    for case in Switch(bits_value):

        # TR and RF bits are low, so we are unknown, between sequences or between pulses.
        # Valid previous states are: unknown and tr_post_pulse
        if case(0):
            if state == states[0]:  # previous state unknown, so leave it that way
                # state = states[1]  # between_sequences
                # pulse_num = 0
                # else:
                #    break
                # if __debug__:
                #    print("Starting sequence")
                continue
            elif (
                state == states[4]
            ):  # previous state tr_post_pulse, now between pulses/sequences
                if pulse_num == num_pulses:  # If we finished the last pulse
                    state = states[1]  # Now we're between sequences
                    if __debug__:
                        print("Starting downtime between sequences")
                    pulse_num = 0
                else:
                    state = states[5]  # Otherwise we're between pulses
                    if __debug__:
                        print("Between pulses")
                timediff = timestamp - timestamp_previous
                time_error_us = (timediff - tr_post_delay_us / 1000000.0) * 1000000.0
                if abs(time_error_us) > THRESHOLD_US:
                    if max_time_error_us < time_error_us:
                        max_time_error_us = time_error_us
                    print(
                        "TIMING ERROR TR_POST_DELAY: "
                        + str(time_error_us)
                        + " us. Resetting"
                    )
                    print("Timestamp: {0} row: {1}".format(timestamp, row_num))
                    state = states[0]
                    pulse_num = 0
            else:
                print(
                    "ERROR WITH PULSE SEQUENCE, state was "
                    + state
                    + " but we now see between pulses or sequences. Resetting"
                )
                print("Timestamp: {0} row: {1}".format(timestamp, row_num))
                state = states[0]  # unknown
                pulse_num = 0
            break

        # We have just the TR bit high
        # Valid previous states are unknown, between_sequences, pulse, or between_pulses
        # Valid states now are tr_pre_pulse or tr_post_pulse
        if case(1):
            if (
                state == states[0]
            ):  # If we are unknown, wait until we've reached a new int.period
                if (timestamp - timestamp_previous) > between_ave_periods_s:
                    # Now at the beginning of a new pulse sequence
                    state = states[2]
                    if __debug__:
                        print("Starting sequence")
                else:  # Otherwise continue because we still don't know where we are
                    continue
            elif (
                state == states[1]
            ):  # If we were between sequences, now we're starting sequence
                # TODO: We can calculate time between sequences here if we want
                # Starting pulse sequence
                state = states[2]  # tr_pre_pulse
            elif state == states[3]:  # If we were in a pulse, now we're post pulse
                timediff = timestamp - timestamp_previous
                time_error_us = (timediff - rf_pulse_time_us / 1000000.0) * 1000000.0
                if abs(time_error_us) > THRESHOLD_US:
                    if max_time_error_us < time_error_us:
                        max_time_error_us = time_error_us
                    print("TIMING ERROR RF PULSE LENGTH: " + str(time_error_us) + " us")
                    print("Timestamp: {0} row: {1}".format(timestamp, row_num))
                    print("Time diff: " + str(timediff * 1000000.0) + " us")
                    state = states[0]
                    pulse_num = 0
                    break
                if (
                    pulse_num < num_pulses
                ):  # If we haven't finished sequence, increment counter
                    pulse_num += 1
                state = states[4]  # tr_post_pulse now
            elif (
                state == states[5]
            ):  # If we were between pulses, should be tr pre-pulse now
                timediff = timestamp - timestamp_previous
                total_pulse_time = tr_post_delay_us + tr_pre_delay_us + rf_pulse_time_us
                inter_pulse_time = tau_us * (
                    pulse_sequence[pulse_num] - pulse_sequence[pulse_num - 1]
                )
                correct_delay_us = inter_pulse_time - total_pulse_time
                time_error_us = (timediff - correct_delay_us / 1000000.0) * 1000000.0
                if abs(time_error_us) > THRESHOLD_US:
                    if max_time_error_us < time_error_us:
                        max_time_error_us = time_error_us
                    print(
                        "TIMING ERROR BETWEEN_PULSES: "
                        + str(time_error_us)
                        + " us. Resetting"
                    )
                    print("Timestamp: {0} row: {1}".format(timestamp, row_num))
                    state = states[0]
                    pulse_num = 0
                    break
                state = states[2]  # tr_pre_pulse
            else:
                print(
                    "ERROR WITH PULSE SEQUENCE, state was "
                    + state
                    + " but we now see only TR high. Resetting"
                )
                print("Timestamp: {0} row: {1}".format(timestamp, row_num))
                state = states[0]
                pulse_num = 0
            break

        # We have just the RF bit high, This is an error condition
        if case(2):
            # Error, RF high but TR is not
            print("ERROR, only RF is high, TR is not!")
            print("Timestamp: {0} row: {1}".format(timestamp, row_num))
            state = states[0]
            pulse_num = 0
            break

        # We have the TR and RF bits high, so we are in state 'pulse'
        # Valid previous states are unknown, or tr_pre_pulse
        # Valid states now are pulse
        if case(3):
            if (
                state == states[0]
            ):  # If we are unknown state just skip this and continue
                continue
            elif state == states[2]:  # If we were in tr_pre_pulse
                timediff = timestamp - timestamp_previous
                time_error_us = (timediff - tr_pre_delay_us / 1000000.0) * 1000000.0
                if abs(time_error_us) > THRESHOLD_US:
                    if max_time_error_us < time_error_us:
                        max_time_error_us = time_error_us
                    print(
                        "TIMING ERROR TR_PRE_DELAY: "
                        + str(time_error_us)
                        + " us. Resetting"
                    )
                    print("Timestamp: {0} row: {1}".format(timestamp, row_num))
                    state = states[0]
                    pulse_num = 0
                    break
                state = states[3]
            else:
                print(
                    "ERROR WITH PULSE SEQUENCE, state was "
                    + state
                    + " but we now see TR and RF high. Resetting"
                )
                print("Timestamp: {0} row: {1}".format(timestamp, row_num))
                state = states[0]
                pulse_num = 0
            break

print("Max timing error: " + str(max_time_error_us) + " us")
