===========
Lab Testing
===========

It is important to verify that the system is operating nominally before deployment and regular operations.
We recommend that you run at least the tests below.

GPS and Reference signal tests
------------------------------

#. Test for GPS lock on the GPS Octoclock.

- Connect a GPS antenna to the Octoclock and place the antenna in a location where it can receive a
  good signal, such as near a window.
- Plug in the power connector for the Octoclock.
- Ensure that the switch of the left side of the Octoclock face is switched to "internal" reference
  this indicates that the Octoclock is generating its own 10 MHz and 1 PPS signals, not being disciplined by external signals.
- Wait for a while until the green "GPS Lock" LED on the front face of the Octoclock is lit. This
  may take up to a half hour. Alternatively, you can write a script using the UHD API to query the device for GPS lock. See
  https://files.ettus.com/manual/classuhd_1_1usrp__clock_1_1multi__usrp__clock.html for more details on how to do
  this.

#. Test synchronicity of multiple Octoclock arrangement.

    * Set the GPS Octoclock to internal reference, and the other two Octoclocks to external reference - these will be
      disciplined by the GPS octoclock.
    * Using cables of equal electrical length, connect the input 10 MHz and input 1 PPS channels for each non-GPS
      octoclock to output 10 MHz and 1 PPS channels of the GPS Octoclock. Equal electrical length cables are required
      to keep the non-GPS Octoclocks synced as closely as possible.
    * Use an oscilloscope to compare 10 MHz and 1 PPS channels between the two disciplined Octoclocks, and between
      output channels on the same Octoclock. Ideally all channels should be no more than 10 nanoseconds different.

#. Test that N200 REF and PPS LEDs are operating correctly.



#. TXIO board testing - TODO: move to here from hardware

#. TODO: Should add blurb here about matching transmitter design

#. Test bandwidth and amplitude of TX waveforms from N200s across the output band

#. Timing stability of GPIO from TXIO board in N200s - pulse width, ATR width, stability in timing between the two

    * Use Logic analyzer to measure this over a long time period, days
    * Can use Log-Amp circuit to give TTL signal (same as the EPOP trigger) when high output power detected (i.e. TX on)
       and measure relative timing between TX and ATR, and widths of both signals.

#. Loopback tests at boresight - allows you to see the differences in channel power after digitizing
   - verifies that N200's are synchronized, this is due to boresight steering having no phase offset
   so with all equal cable lengths the phase output on all antennas should be the same (check with
   rawrf and antennas_iq) - scripts are available under
   ``tools/testing_utils/plot_borealis_hdf5_data/``

#. Logic analyzer tests with a transmitter

#. Long-term reliability tests of software

#. Scope output tests - verify pulse shape of TX out - verify GPIO signals (T/R) - verify pulse
   distances


