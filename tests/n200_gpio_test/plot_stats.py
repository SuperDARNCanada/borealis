#!/bin/env python

import matplotlib.pyplot as plt

import numpy

data = numpy.loadtxt("stats_20161011_us.txt", skiprows=1)
x0 = numpy.arange(0, len(data[:, 0]))
x1 = numpy.arange(0, len(data[:, 1]))
x2 = numpy.arange(0, len(data[:, 2]))

y0 = data[:, 0]
y1 = data[:, 1]
y2 = data[:, 2]

fig, ax = plt.subplots()

plt.plot(x0, y0, "r--o", label="set_time_now")
plt.plot(x1, y1, "c-D", label="set_command_time")
plt.plot(x2, y2, "y-8", label="set_gpio_attribute")
plt.ylabel("Time (us)")
plt.xlabel("Run")
plt.legend()
fig.savefig("stats.png")
