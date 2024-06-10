#!/usr/bin/env python3
# Note that this script works with the new generation of Saleae logic analyzers only
import saleae

s = saleae.Saleae()
min_sample_rate = 1e6
channels = [0, 1, 2, 3]
channel_triggers = []
for c in range(0, 8):
    channel_triggers.append(saleae.Trigger.Posedge)

performance_value = saleae.PerformanceOption.Full
print("Connected devices: " + str(s.get_connected_devices()))

# s.set_active_channels(channels) # Cannot do this with old Logic 8
# s.set_triggers_for_all_channels(channel_triggers)
# s.set_num_samples(10e6)
s.set_capture_seconds(3.5)
s.set_sample_rate_by_minimum(min_sample_rate)
# s.set_performance(performance_value)
# print("USB Bandwidth: " + str(s.get_bandwidth(min_sample_rate)))
print("Performance: " + str(s.get_performance()))
# print("Active devices: " + str(s.get_active_devices()))
print("Active channels: " + str(s.get_active_channels()))
