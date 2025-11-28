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
of bandpass filters (sets of coefficients) required is defined as :math:`\color{lightgray}P`, and this value is also,
therefore, the number of unique :math:`\color{lightgray}{\phi}` as shown below. The method can really be defined as
doing the following:

.. math::

   \color{lightgray}{b}_k[n] = h[n]e^{j({\phi}_k + 2{\pi}n\frac{f}{{F}_s})}

where :math:`\color{lightgray}{b}_k` are the bandpass filters from :math:`\color{lightgray}k=0` to :math:`\color{lightgray}k=P`. :math:`\color{lightgray}{h[n]}` is the
original low pass filter coefficient set of length :math:`\color{lightgray}N`, :math:`\color{lightgray}f` is the translation
frequency, and :math:`\color{lightgray}{F}_s` is the input sampling frequency. :math:`\color{lightgray}{{\phi}_k}` is the starting
phase of the NCO (numerically controlled oscillator) being multiplied element by element with the
low pass filter where

.. math:: \color{lightgray}{\phi}_k = 2{\pi}Rk{\frac{f}{{F}_s}}

and where the minimum integer value :math:`\color{lightgray}P` is determined by the equation given by Frerking:

.. math:: \color{lightgray}PR\frac{f}{{F}_s} = int,\ \ 1 \leq P \leq {F}_s

where :math:`\color{lightgray}R` is the integer decimation rate. The maximum value of :math:`\color{lightgray}P` would then be
:math:`\color{lightgray}{F}_s`, assuming :math:`\color{lightgray}f` and :math:`\color{lightgray}{F}_s` are integers.

Then, to filter and decimate,

.. math:: \color{lightgray}{y[m]} = {y[Rl]} = \sum\limits_{n=0}^N x[Rl-n]{b}_{(n{\bmod}P)}[n]

where :math:`\color{lightgray}{y[m]}` is each baseband decimated sample, and :math:`\color{lightgray}{x[l]}` is the input samples. By
decimation, the output number of samples, :math:`\color{lightgray}M = \frac{L}{R}` where :math:`\color{lightgray}L` is the input
number of samples (although to avoid zero-padding for convolution, :math:`\color{lightgray}M< {\frac{L}{R}}` ).

Our new sampling rate will be

.. math:: \color{lightgray}{F}_{new} = \frac{{F}_{s}}{R}

However, by using a single bandpass filter, a new method could be used. The starting phase of the
NCO on the filter coefficient set is pulled out from the sum, and then phase correction is done on
the decimated samples after the convolution step.

.. math:: \color{lightgray}{{b}[n]} = h[n]e^{j({2{\pi}n\frac{f}{{F}_s}})}

.. math:: \color{lightgray}{y[m]} = {y[Rl]} = e^{j{\phi}_k} \sum\limits_{n=0}^N x[Rl-n]{b[n]},\ \ k = m{\bmod}P

Both methods are equivalent:

.. math:: \color{lightgray}e^{j{\phi}_k} \sum\limits_{n=0}^N x[Rl-n]h[n]e^{j(2{\pi}n\frac{f}{{F}_s})} = \sum\limits_{n=0}^N x[Rl-n]h[n]e^{j({\phi}_k + 2{\pi}n\frac{f}{{F}_s})}

Frerking’s method requires :math:`\color{lightgray}NP` multiplications before convolution, and for it to be most
computationally efficient, it requires storing :math:`\color{lightgray}P` sets of :math:`\color{lightgray}N` coefficients. For a small
value of :math:`\color{lightgray}P` and a large value of :math:`\color{lightgray}M` output samples, the number of multiplications
would be minimized by this method. However, the worst case for using Frerking’s method is a large
value of :math:`\color{lightgray}{F}_s`, :math:`\color{lightgray}M \ge {F}_s`, and an unknown :math:`\color{lightgray}f`, meaning that the storage
requirements would be for :math:`\color{lightgray}P = {F}_s` number of sets of filter coefficients.

For the case when there exists a small value of :math:`\color{lightgray}M` or a large value of :math:`\color{lightgray}P` or
:math:`\color{lightgray}N`, the new modified method might be more computationally efficient, as :math:`\color{lightgray}N + M -
\lfloor {\frac{M}{P}} \rfloor` multiplications are required in this method. However, the new method
is more memory efficient in all cases where :math:`\color{lightgray}P > 1` because only one set of filter
coefficients is required to be stored in all cases.

For an unknown integer value :math:`\color{lightgray}f` and an unknown decimation rate (or where :math:`\color{lightgray}R` is not a
submultiple of :math:`\color{lightgray}{F}_s`), processing would have to accommodate :math:`\color{lightgray}P = {F}_s`, and so
Frerking would be optimal where

.. math:: \color{lightgray}N{F}_s < N + M - \lfloor{\frac{M}{{F}_s}}\rfloor

and the new method would be optimal for

.. math:: \color{lightgray}N{F}_s > N + M - \lfloor{\frac{M}{{F}_s}}\rfloor
