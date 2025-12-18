:orphan:

.. _frerking:

-------------------------------------------
Another representation of Frerking’s method
-------------------------------------------

Frerking’s method is found in Frerking, M. E., *Digital Signal Processing in Communications
Systems*, Chapman & Hall, 1994, pp. 171-174. It is a method for creating a frequency-translating FIR
filter by translating the filter coefficients to a bandpass filter and then convolving with the
input samples (to simultaneously mix to baseband and decimate). The method involves creating
multiple bandpass filters so as to maintain the linear phase property of the FIR filter. The number
of bandpass filters (sets of coefficients) required is defined as :math:`\color{Gray}P`, and this value is also,
therefore, the number of unique :math:`\color{Gray}{\phi}` as shown below. The method can really be defined as
doing the following:

.. math::

   \color{Gray}{b}_k[n] = h[n]e^{j({\phi}_k + 2{\pi}n\frac{f}{{F}_s})}

where :math:`\color{Gray}{b}_k` are the bandpass filters from :math:`\color{Gray}k=0` to :math:`\color{Gray}k=P`. :math:`\color{Gray}{h[n]}` is the
original low pass filter coefficient set of length :math:`\color{Gray}N`, :math:`\color{Gray}f` is the translation
frequency, and :math:`\color{Gray}{F}_s` is the input sampling frequency. :math:`\color{Gray}{{\phi}_k}` is the starting
phase of the NCO (numerically controlled oscillator) being multiplied element by element with the
low pass filter where

.. math:: \color{Gray}{\phi}_k = 2{\pi}Rk{\frac{f}{{F}_s}}

and where the minimum integer value :math:`\color{Gray}P` is determined by the equation given by Frerking:

.. math:: \color{Gray}PR\frac{f}{{F}_s} = int,\ \ 1 \leq P \leq {F}_s

where :math:`\color{Gray}R` is the integer decimation rate. The maximum value of :math:`\color{Gray}P` would then be
:math:`\color{Gray}{F}_s`, assuming :math:`\color{Gray}f` and :math:`\color{Gray}{F}_s` are integers.

Then, to filter and decimate,

.. math:: \color{Gray}{y[m]} = {y[Rl]} = \sum\limits_{n=0}^N x[Rl-n]{b}_{(n{\bmod}P)}[n]

where :math:`\color{Gray}{y[m]}` is each baseband decimated sample, and :math:`\color{Gray}{x[l]}` is the input samples. By
decimation, the output number of samples, :math:`\color{Gray}M = \frac{L}{R}` where :math:`\color{Gray}L` is the input
number of samples (although to avoid zero-padding for convolution, :math:`\color{Gray}M< {\frac{L}{R}}` ).

Our new sampling rate will be

.. math:: \color{Gray}{F}_{new} = \frac{{F}_{s}}{R}

However, by using a single bandpass filter, a new method could be used. The starting phase of the
NCO on the filter coefficient set is pulled out from the sum, and then phase correction is done on
the decimated samples after the convolution step.

.. math:: \color{Gray}{{b}[n]} = h[n]e^{j({2{\pi}n\frac{f}{{F}_s}})}

.. math:: \color{Gray}{y[m]} = {y[Rl]} = e^{j{\phi}_k} \sum\limits_{n=0}^N x[Rl-n]{b[n]},\ \ k = m{\bmod}P

Both methods are equivalent:

.. math:: \color{Gray}e^{j{\phi}_k} \sum\limits_{n=0}^N x[Rl-n]h[n]e^{j(2{\pi}n\frac{f}{{F}_s})} = \sum\limits_{n=0}^N x[Rl-n]h[n]e^{j({\phi}_k + 2{\pi}n\frac{f}{{F}_s})}

Frerking’s method requires :math:`\color{Gray}NP` multiplications before convolution, and for it to be most
computationally efficient, it requires storing :math:`\color{Gray}P` sets of :math:`\color{Gray}N` coefficients. For a small
value of :math:`\color{Gray}P` and a large value of :math:`\color{Gray}M` output samples, the number of multiplications
would be minimized by this method. However, the worst case for using Frerking’s method is a large
value of :math:`\color{Gray}{F}_s`, :math:`\color{Gray}M \ge {F}_s`, and an unknown :math:`\color{Gray}f`, meaning that the storage
requirements would be for :math:`\color{Gray}P = {F}_s` number of sets of filter coefficients.

For the case when there exists a small value of :math:`\color{Gray}M` or a large value of :math:`\color{Gray}P` or
:math:`\color{Gray}N`, the new modified method might be more computationally efficient, as :math:`\color{Gray}N + M -
\lfloor {\frac{M}{P}} \rfloor` multiplications are required in this method. However, the new method
is more memory efficient in all cases where :math:`\color{Gray}P > 1` because only one set of filter
coefficients is required to be stored in all cases.

For an unknown integer value :math:`\color{Gray}f` and an unknown decimation rate (or where :math:`\color{Gray}R` is not a
submultiple of :math:`\color{Gray}{F}_s`), processing would have to accommodate :math:`\color{Gray}P = {F}_s`, and so
Frerking would be optimal where

.. math:: \color{Gray}N{F}_s < N + M - \lfloor{\frac{M}{{F}_s}}\rfloor

and the new method would be optimal for

.. math:: \color{Gray}N{F}_s > N + M - \lfloor{\frac{M}{{F}_s}}\rfloor
