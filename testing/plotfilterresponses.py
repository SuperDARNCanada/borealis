from scipy import signal
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft,fftshift
import sys

filename = sys.argv[1]

with open(filename,'r') as f:
    filter_coeff = f.read()

filter_coeff = [float(h) for h in filter_coeff.splitlines()]
w,h = signal.freqz(filter_coeff, whole=True)

fig = plt.figure()
plt.title('Digital filter frequency response')
ax1 = fig.add_subplot(111)
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.ylabel('Amplitude [dB]', color='b')
plt.xlabel('Frequency [rad/sample]')
ax2 = ax1.twinx()
angles = np.unwrap(np.angle(h))
plt.plot(w, angles, 'g')
plt.ylabel('Angle (radians)', color='g')
plt.grid()
plt.axis('tight')

plt.show()