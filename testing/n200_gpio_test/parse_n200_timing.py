#! /usr/bin/env python

# Nabbed this class from http://code.activestate.com/recipes/410692/
# This class provides the functionality we want. You only need to look at
# this if you want to know how this works. It only needs to be defined
# once, no need to muck around with its internals.


class switch(object):
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


# This script takes in a csv exported file from the Saleae Logic software
# and finds out if the timing passes or fails 
import csv
import sys

# Columns in the csv file correspond to certain information. 
# There is a timestamp, a scope sync bit, an atten bit, a tr bit and an rf bit.
##time_index = 0
##ss_index = 1
##a_index = 2
##tr_index = 3
##rf_index = 4

# This is for the ATR version of the borealis code where we have a SS bit and an ATR bit only
time_index = 0
ss_index = 1
atr_index = 2

# The time delays between each of the various events can be programmed, so here we can change them
# start_delay_us = 49500            # Delay between scope sync low and scope sync high
# scope_sync_pre_delay_us = 0  # Delay between scope sync high and ATTEN high
# atten_pre_delay_us = 10       # Delay between ATTEN high and TR high
# tr_pre_delay_us = 60           # Delay between TR high and RF pulse high
# rf_pulse_time_us = 300        # How long is the RF pulse?
# tr_post_delay_us = 40         # Delay between RF low and TR low
# atten_post_delay_us = 10       # Delay between TR low and ATTEN low
THRESHOLD_US = 1
BETWEEN_PULSE_SEQUENCES_THRESHOLD_US = 4000  # threshold for between pulse sequences
# scope_sync_pre_delay_to_atr_us = 0 # how long between scope sync high and first atr going high?
atr_pulse_length_us = 320  # How long is the atr pulse ?

# Various items related to scope sync delay
num_ranges = 75
rsep_km = 45
first_range_km = 180
speed_of_light = 299792458.0
scope_sync_post_delay_us = (num_ranges * rsep_km + first_range_km) * \
                           1000.0 * 1000000 * 2.0 / speed_of_light
print("Scope sync post delay: " + str(scope_sync_post_delay_us) + " us")

# Items related to the pulse sequence
tau_us = 1500
# pulse_sequence = [0,14,22,24,27,31,42,43]
# pulse_sequence = [0,5,8,12,18,30,32,33]
pulse_sequence_8 = [0, 14, 22, 24, 27, 31, 42, 43]
pulse_sequence = pulse_sequence_8

# State machine
# states = ("unknown", "between_sequences", "start_of_sequence", "atten_pre_pulse", "tr_pre_pulse", "pulse", "tr_post_pulse", "atten_post_pulse", "between_pulses", "end_of_sequence")
states = ("unknown", "between_sequences", "start_of_sequence", "atr_pulse", "between_pulses",
          "end_of_sequence")  # For ATR
state = states[0]
print("state is : " + state)

# Get the file and read into memory
test_file = csv.reader(open(sys.argv[1], 'r'), delimiter=',')

max_time_error_us = 0.0
timestamp = 0.0
pulse_num = 0
num_pulses = len(pulse_sequence)
for row in test_file:
    timestamp_previous = timestamp
    timestamp = float(row[0])
    ## There are 16 possible cases for the 4 bits, so let's just handle them all
    # bits_value = (int(row[ss_index]) << ss_index-1)
    # bits_value += (int(row[a_index]) << a_index-1)
    # bits_value += (int(row[tr_index]) << tr_index-1)
    # bits_value += (int(row[rf_index]) << rf_index-1)
    # There are 4 possible cases for the 2 bits, so handle them all
    bits_value = (int(row[ss_index]) << ss_index - 1)
    bits_value += (int(row[atr_index]) << atr_index - 1)
    print("state: " + state + " pulse number: " + str(pulse_num))
    print("Bits value : " + str(bits_value))
    for case in switch(bits_value):
        if case(0):  # Both ss and atr are low, so we are either at start of sequence or between pulses...
            if state == states[0]:  # if we are currently in an unknown state, then let's just continue
                continue
            elif state == states[1]:  # if we were previously between sequences, then we are now at start of sequence
                state = states[2]
                pulse_num = 0
                timediff = timestamp - timestamp_previous
                if abs(timediff) > BETWEEN_PULSE_SEQUENCES_THRESHOLD_US:
                    print "TIMING ERROR: LARGE TIME BETWEEN SEQUENCES: " + str(timediff) + " us"
            elif state == states[3]:  # previously in atr pulse, now between pulses
                timediff = timestamp - timestamp_previous
                time_error_us = (timediff - atr_pulse_length_us / 1000000.0) * 1000000.0
                if abs(time_error_us) > THRESHOLD_US:
                    if max_time_error_us < time_error_us:
                        max_time_error_us = time_error_us
                    print("TIMING ERROR: ATR PULSE LENGTH: " + str(time_error_us) + " us")
                    print("Time diff: " + str(timediff * 1000000.0) + " us")
                if pulse_num == num_pulses - 1:
                    state = states[5]
                else:
                    pulse_num += 1
                    # Between pulses
                    state = states[4]
            else:
                print ("ERROR WITH PULSE SEQUENCE, state was " + state +
                       " but we now see between pulses or beginning of sequence. Resetting")
                state == states[0]
            break

        if case(1):  # Between sequences
            if state == states[0]:
                state = states[1]  # Now between sequences if we didn't know before
            elif state == states[5]:  # End of sequence, so now between sequences
                timediff = timestamp - timestamp_previous
                time_error_us = (timediff - scope_sync_post_delay_us / 1000000.0) * 1000000.0
                if abs(time_error_us) > THRESHOLD_US:
                    if max_time_error_us < time_error_us:
                        max_time_error_us = time_error_us
                    print("TIMING ERROR SS_POST_DELAY: " + str(time_error_us) + " us")
                    state = states[1]
            else:
                print ("ERROR WITH PULSE SEQUENCE, state was " + state +
                       " but we now see only ss high. Resetting")
                state = states[0]
            break

        if case(2):  # ATR high, ss low, so in a pulse
            if state == states[0]:
                continue
            elif state == states[1]:  # between sequences, so first pulse now
                if pulse_num != 0:
                    print("ERROR: We were between sequences, but pulse number is not 0! Resetting")
                    state = states[0]
                    break
                timediff = timestamp - timestamp_previous
                if abs(timediff) > BETWEEN_PULSE_SEQUENCES_THRESHOLD_US:
                    print "TIMING ERROR: LARGE TIME BETWEEN SEQUENCES: " + str(timediff) + " us"
                pulse_num += 1
                state = states[3]  # Still in a pulse
            elif state == states[4]:  # Between pulses, so in some pulse now
                if pulse_num == num_pulses - 1:
                    state = states[5]  # now in end of sequence
                else:
                    pulse_num += 1
                    state = states[3]  # Still in a pulse
                    timediff = timestamp - timestamp_previous
                    correct_delay_us = tau_us * (pulse_sequence[pulse_num] -
                                                 pulse_sequence[pulse_num - 1]) - \
                                       atr_pulse_len_us
                    time_error_us = (timediff - correct_delay_us / 1000000.0) * 1000000.0
                    if abs(time_error_us) > THRESHOLD_US:
                        if max_time_error_us < time_error_us:
                            max_time_error_us = time_error_us
                        print("TIMING ERROR between pulses: " + str(time_error_us) + " us")
            else:
                print("ERROR WITH PULSE SEQUENCE, state was " + state +
                      " but we now see both atr and ss high. Resetting")
                state = states[0]
                break

        if case(3):
            print ("ERROR: ATR and SS both HIGH. Resetting")
            state = states[0]
            break
            # atr and ss high,  error
            # if state == states[0]:
            #  continue
            # elif state == states[2]:
            #  timediff = timestamp - timestamp_previous
            #  time_error_us = (timediff-scope_sync_pre_delay_us/1000000.0)*1000000.0
            #  if(abs(time_error_us) > THRESHOLD_US):
            #    if max_time_error_us < time_error_us:
            #      max_time_error_us = time_error_us
            #    print("TIMING ERROR1: " + str(time_error_us) + " us")
            #  # Pre pulse
            #  state = states[3]
            # elif state == states[8]:
            #  timediff = timestamp - timestamp_previous
            #  correct_delay_us = tau_us*(pulse_sequence[pulse_num]-pulse_sequence[pulse_num-1]) - tr_post_delay_us - atten_post_delay_us - atten_pre_delay_us - tr_pre_delay_us - rf_pulse_time_us
            #  time_error_us = (timediff - correct_delay_us/1000000.0)*1000000.0
            #  if(abs(time_error_us) > THRESHOLD_US):
            #    if max_time_error_us < time_error_us:
            #      max_time_error_us = time_error_us
            #    print("TIMING ERROR2: " + str(time_error_us) + " us")
            #  # Pre pulse
            #  state = states[3]
            # elif state == states[6]:
            #  timediff = timestamp - timestamp_previous
            #  time_error_us = (timediff-tr_post_delay_us/1000000.0)*1000000.0
            #  if(abs(time_error_us) > THRESHOLD_US):
            #    if max_time_error_us < time_error_us:
            #      max_time_error_us = time_error_us
            #    print("TIMING ERROR3: " + str(time_error_us) + " us")
            #    print("Time diff: " + str(timediff*1000000.0) + " us")
            #  # Post pulse
            #  state = states[7]
            # else:
            #  print ("ERROR WITH PULSE SEQUENCE, state was "+state+" but we now see atten and ss high. Resetting")
            #  state == states[0]
            # break

            # if case(7):
            #  # Scope sync, atten and tr high - either before or right after a pulse
            #  if state == states[0]:
            #    continue
            #  elif state == states[3]:
            #    timediff = timestamp - timestamp_previous
            #    time_error_us = (timediff - atten_pre_delay_us/1000000.0)*1000000.0
            #    if(abs(time_error_us) > THRESHOLD_US):
            #      if max_time_error_us < time_error_us:
            #        max_time_error_us = time_error_us
            #      print("TIMING ERROR1: " + str(time_error_us) + " us")
            #    state = states[4]
            #  elif state == states[5]:
            #    timediff = timestamp - timestamp_previous
            #    time_error_us = (timediff - rf_pulse_time_us/1000000.0)*1000000.0
            #    if(abs(time_error_us) > THRESHOLD_US):
            #      if max_time_error_us < time_error_us:
            #        max_time_error_us = time_error_us
            #      print("TIMING ERROR2: " + str(time_error_us) + " us")
            #    state = states[6]
            #  else:
            #    print ("ERROR WITH PULSE SEQUENCE, state was "+state+" but we now see ss atten and tr high. Resetting")
            #    state == states[0]
            #  break

            # if case(15):
            #  if state == states[0]:
            #    continue
            #  elif state == states[4]:
            #    timediff = timestamp - timestamp_previous
            #    time_error_us = (timediff - tr_pre_delay_us/1000000.0)*1000000.0
            #    if(abs(time_error_us) > THRESHOLD_US):
            #      if max_time_error_us < time_error_us:
            #        max_time_error_us = time_error_us
            #      print("TIMING ERROR: " + str(time_error_us) + " us")
            #    # We now go from tr high to pulse
            #    state = states[5]
            #  else:
            #    # Error
            #    print("ERROR WITH PULSE SEQUENCE, state was " + state + " but now we see all high. Resetting")
            #    state = states[0]
            #  break
            #
print("Max timing error: " + str(max_time_error_us) + " us")
