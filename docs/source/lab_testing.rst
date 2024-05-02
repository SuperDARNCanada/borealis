.. _lab-testing:

===========
Lab Testing
===========

It is important to verify that the system is operating nominally before deployment and regular
operations. We recommend that you run at least the tests below.

GPS and Reference signal tests
------------------------------

#. Test for GPS lock on the GPS Octoclock.

   - Connect a GPS antenna to the Octoclock and place the antenna in a location where it can receive a
     good signal, such as near a window.
   - Plug in the power connector for the Octoclock.
   - Ensure that the switch of the left side of the Octoclock face is switched to "internal" reference
     this indicates that the Octoclock is generating its own 10 MHz and 1 PPS signals, not being
     disciplined by external signals.
   - Wait for a while until the green "GPS Lock" LED on the front face of the Octoclock is lit. This
     may take up to a half hour. Alternatively, you can write a script using the UHD API to query the
     device for GPS lock. See
     https://files.ettus.com/manual/classuhd_1_1usrp__clock_1_1multi__usrp__clock.html for more details
     on how to do this.

#. Test synchronicity of multiple Octoclock arrangement.

   - Set the GPS Octoclock to internal reference, and the other two Octoclocks to external reference -
     these will be disciplined by the GPS octoclock.
   - Using cables of equal electrical length, connect the input 10 MHz and input 1 PPS channels for
     each non-GPS octoclock to output 10 MHz and 1 PPS channels of the GPS Octoclock. Equal electrical
     length cables are required to keep the non-GPS Octoclocks synced as closely as possible.
   - Use an oscilloscope to compare 10 MHz and 1 PPS channels between the two disciplined Octoclocks,
     and between output channels on the same Octoclock. Ideally all channels should be no more than 10
     nanoseconds different.

#. Test that N200 REF and PPS LEDs are operating correctly.

Transmitter interface testing
-----------------------------

#. TXIO board testing - TODO: move to here from hardware

N200 Output Testing
-------------------

#. Test bandwidth and amplitude of TX waveforms from N200s across the output bandwidth

#. Test Timing stability of GPIO from TXIO board in N200s

#. pulse width

   - ATR width
   - long-term stability in timing. Use a Logic analyzer to measure this over a long time period (days)
   - A Log-Amp circuit can give a TTL signal (same as the EPOP trigger) when a high output power
     is detected from the transmitter (i.e. TX on) and measure relative timing between TX and ATR,
     and widths of both signals.

#. Loopback tests at boresight. This test allows you to see the differences in channel power after
   digitizing. It verifies that N200s are synchronized, this is due to boresight steering having no
   phase offset so with all equal cable lengths the phase output on all antennas should be the same
   (check with rawrf and antennas_iq) - scripts are available under
   ``tools/testing_utils/plot_borealis_hdf5_data/``

#. Long-term reliability tests of software

#. Scope output tests

   - verify pulse shape of TX out
   - verify GPIO signals (T/R)
   - verify pulse distances

.. _Filter Testing:

Filter Testing
--------------

A Jupyter notebook called ``filters.ipynb`` is located in the ``tests/dsp_testing`` directory of Borealis.
This notebook describes in detail the default ``DecimationScheme`` used by Borealis, the helper functions in
``decimation_scheme.py`` for creating a digital filter, and creates an alternative filter with comparison
to the default. This notebook also benchmarks the performance of the filter schemes on the GPU, both in runtime
and in memory usage. Finally, the ramp-up and ramp-down of transmitted pulses is looked at, for characterization
of the expected transmission spectrum.

This notebook is intended to make it easy to design and prototype new filtering schemes, which is useful for
experiments which, for example:

* Have a non-standard pulse length (the default is 300 microseconds)
* Use range gates of differing size (the default is 45 km)
* Use pulse compression (this increases the transmission bandwidth)
* Are listening experiments (e.g. to measure the frequency spectrum)
* Use a different receiver bandwidth (i.e. not the default 5 MHz)

In any of these circumstances, it is important to design a filter which works for the situation at hand.

.. _Config Testing:

Config File Testing
-------------------

A Python ``unittest`` script for verifying config files is located in ``tests/config_files`` directory.

.. automodule:: tests.config_files.config_testing
    :noindex:
    :no-members:

.. _Realtime Testing:

Realtime Data Simulator
-----------------------

A simulator script is available at ``tests/simulators/realtime/realtime_sim.py``. This script calls the
``realtime_server()`` function of the realtime module, and tests sending a rawacf record to it repeatedly.
The script logs each record that it sends, and each fitacf response that it receives.
