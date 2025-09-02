from scipy import signal
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]

with open(filename, "r") as f:
    filter_coeff = f.read()

filter_coeff = [eval(h)[0] for h in filter_coeff.splitlines()]
w, h = signal.freqz(filter_coeff, whole=True)


w = [x - 2 * np.pi if x > np.pi else x for x in w]
halfway = len(w) / 2
if halfway % 2 == 0:
    new_w = np.concatenate((w[halfway:], w[:halfway]))
    new_h = np.concatenate((h[halfway:], h[:halfway]))
else:
    new_w = np.concatenate((w[halfway + 1 :], w[:halfway]))
    new_h = np.concatenate((h[halfway + 1 :], h[:halfway]))

new_w = new_w[1:]
new_w = new_w[1:]
new_h = new_h[1:]
new_h = new_h[1:]

new_w = new_w[:-1]
new_w = new_w[:-1]
new_h = new_h[:-1]
new_h = new_h[:-1]

fig = plt.figure()
plt.title("Digital filter frequency response")
ax1 = fig.add_subplot(111)
plt.plot(new_w, 20 * np.log10(abs(new_h)), "b")
plt.ylabel("Amplitude [dB]", color="b")
plt.xlabel("Frequency [rad/sample]")
ax2 = ax1.twinx()
angles = np.unwrap(np.angle(new_h))
plt.plot(new_w, angles, "g")
plt.ylabel("Angle (radians)", color="g")
plt.grid()
plt.axis("tight")

plt.show()
plt.save(filename + ".png")
