.. _frerking-label:

-------------------------------------------
Another representation of Frerking’s method
-------------------------------------------

Frerking’s method is found in Frerking, M. E., *Digital Signal Processing in Communications
Systems*, Chapman & Hall, 1994, pp. 171-174. It is a method for creating a frequency-translating FIR
filter by translating the filter coefficients to a bandpass filter and then convolving with the
input samples (to simultaneously mix to baseband and decimate). The method involves creating
multiple bandpass filters so as to maintain the linear phase property of the FIR filter. The number
of bandpass filters (sets of coefficients) required is defined as :math:`P`, and this value is also,
therefore, the number of unique :math:`{\phi}` as shown below. The method can really be defined as
doing the following:

.. math::

   \label{eq1}
               {{b}_k[n]} = h[n]e^{j({\phi}_k + 2{\pi}n\frac{f}{{F}_s})}

where :math:`{b}_k` are the bandpass filters from :math:`k=0` to :math:`k=P`. :math:`{h[n]}` is the
original low pass filter coefficient set of length :math:`N`, :math:`f` is the translation
frequency, and :math:`{F}_s` is the input sampling frequency. :math:`{{\phi}_k}` is the starting
phase of the NCO (numerically controlled oscillator) being multiplied element by element with the
low pass filter where

.. math:: {\phi}_k = 2{\pi}Rk{\frac{f}{{F}_s}}

and where the minimum integer value :math:`P` is determined by the equation given by Frerking:

.. math:: PR\frac{f}{{F}_s} = int,\ \ 1 \leq P \leq {F}_s

where :math:`R` is the integer decimation rate. The maximum value of :math:`P` would then be
:math:`{F}_s`, assuming :math:`f` and :math:`{F}_s` are integers.

Then, to filter and decimate,

.. math:: {y[m]} = {y[Rl]} = \sum\limits_{n=0}^N x[Rl-n]{b}_{(n{\bmod}P)}[n]

where :math:`{y[m]}` is each baseband decimated sample, and :math:`{x[l]}` is the input samples. By
decimation, the output number of samples, :math:`M = \frac{L}{R}` where :math:`L` is the input
number of samples (although to avoid zero-padding for convolution, :math:`M< {\frac{L}{R}}` ).

Our new sampling rate will be

.. math:: {F}_{new} = \frac{{F}_{s}}{R}

However, by using a single bandpass filter, a new method could be used. The starting phase of the
NCO on the filter coefficient set is pulled out from the sum, and then phase correction is done on
the decimated samples after the convolution step.

.. math:: {{b}[n]} = h[n]e^{j({2{\pi}n\frac{f}{{F}_s}})}

.. math:: {y[m]} = {y[Rl]} = e^{j{\phi}_k} \sum\limits_{n=0}^N x[Rl-n]{b[n]},\ \ k = m{\bmod}P

Both methods are equivalent:

.. math:: e^{j{\phi}_k} \sum\limits_{n=0}^N x[Rl-n]h[n]e^{j(2{\pi}n\frac{f}{{F}_s})} = \sum\limits_{n=0}^N x[Rl-n]h[n]e^{j({\phi}_k + 2{\pi}n\frac{f}{{F}_s})}

Frerking’s method requires :math:`NP` multiplications before convolution, and for it to be most
computationally efficient, it requires storing :math:`P` sets of :math:`N` coefficients. For a small
value of :math:`P` and a large value of :math:`M` output samples, the number of multiplications
would be minimized by this method. However, the worst case for using Frerking’s method is a large
value of :math:`{F}_s`, :math:`M \ge {F}_s`, and an unknown :math:`f`, meaning that the storage
requirements would be for :math:`P = {F}_s` number of sets of filter coefficients.

For the case when there exists a small value of :math:`M` or a large value of :math:`P` or
:math:`N`, the new modified method might be more computationally efficient, as :math:`N + M -
\lfloor {\frac{M}{P}} \rfloor` multiplications are required in this method. However, the new method
is more memory efficient in all cases where :math:`P > 1` because only one set of filter
coefficients is required to be stored in all cases.

For an unknown integer value :math:`f` and an unknown decimation rate (or where :math:`R` is not a
submultiple of :math:`{F}_s`), processing would have to accommodate :math:`P = {F}_s`, and so
Frerking would be optimal where

.. math:: N{F}_s < N + M - \lfloor{\frac{M}{{F}_s}}\rfloor

and the new method would be optimal for

.. math:: N{F}_s > N + M - \lfloor{\frac{M}{{F}_s}}\rfloor
