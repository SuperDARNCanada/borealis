.. _hardware-verification:

=====================
Hardware Verification
=====================

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

Follow the testing procedure below to run a simple test of the TXIO inputs and outputs.
There are two types of tests, a single ended output test which tests the SMA outputs and LEDs,
and a loopback test which tests the differential signal outputs and inputs without an expensive
differential probe. Reference the above image of the rear of the N200 for pinouts.

- Connect a needle probe to channel one of your oscilloscope and set it to trigger on the
  rising edge of channel one.
- Connect a needle probe to channel two of your oscilloscope, to be used in later tests.
- Run ``test_txio_gpio.py`` located in ``borealis/tests/test_rx_tx/test_txio_gpio/``. Usage is
  as follows (assuming the default IP address)::

     python3 test_txio_gpio.py 192.168.10.2

  When prompted to enter the pins corresponding to the TXIO signals, press enter to accept the
  default pin settings. This will begin the tests.

#. Insert the needle probe into the SMA output corresponding to ``RXo``, this should be the
   right-most SMA output when facing the N200 from the back.

   - Verify that the GREEN LED is flashing, and all others are unlit.
   - Verify that the scope signal is the inverse of the pattern flashed by the GREEN front LED.
   - Proceed to the next test (CTRL+C, then enter "y").

#. Insert the needle probe into the SMA output corresponding to ``TXo``, this should be the second
   SMA output from the left when facing the N200 from the back.

   - Verify that the RED and BLUE LEDs are flashing together, and both others are unlit.
   - Verify that the scope signal is the inverse of the pattern flashed by the RED and BLUE front LEDs.
   - Proceed to the next test (CTRL+C, then enter "y").

#. Insert the needle probe into the SMA output corresponding to ``TR``, this should be the left-most
   SMA output when facing the N200 from the back.

   .. note::

      You will not move on to the next test until you verify the SMA ``TR``, ``TR+``, and ``TR-`` signals on
      the oscilloscope.

   - Verify that the BLUE and GREEN LEDs are flashing together, and both others are unlit.
   - Verify that the scope signal is the inverse of the pattern flashed by the BLUE and GREEN front LEDs.

   Insert the needle probe into the hole corresponding to pin 7 of the D-Sub connector (``TR+``,
   yellow wire, J2 pin 10).

   - Verify that the scope signal is following the pattern flashed by the BLUE and GREEN front LEDs.

   Insert the needle probe into the hole corresponding to pin 2 of the D-Sub connector (``TR-``,
   orange wire, J2 pin 8).

   - Verify that the scope signal is the inverse of the pattern flashed by the BLUE and GREEN front LEDs.
   - Proceed to the next test (CTRL+C, then enter "y").

#. Insert the needle probe into SMA output corresponding to ``IDLE``, this should be the third SMA
   output from the left when facing the N200 from the back.

   - Verify that the YELLOW LED is flashing, and all others are unlit.
   - Verify that the scope signal is the inverse of the pattern flashed by the YELLOW front LED.
   - Proceed to the next test (CTRL+C, then enter "y").

#. Insert the needle probe into the hole corresponding to pin 8 of the D-Sub (``TM+``, green wire, J2 pin 4)
   Insert the needle probe from the oscilloscope channel two into the hole corresponding to pin 3 of the D-Sub
   (``TM-``, blue wire, J2 pin 2).

   - Verify that the scope signals for channel 1 and 2 are showing opposing pulses approximately 1 second in
     width, with a 2 second period (50% duty cycle). In other words, they are 180 degrees out of phase.
   - Proceed to the next test (CTRL+C, then enter "y").

#. To properly perform the loopback tests of the differential signals, connect the D-Sub pins to
   each other in the following configuration:

   - Pin 6 to pin 7 - ``AGC+`` to ``TR+``, Red wire to Yellow wire
   - Pin 1 to pin 2 - ``AGC-`` to ``TR-``, Brown wire to Orange wire
   - Pin 8 to pin 9 - ``TM+`` to ``LP+``, Green wire to Purple wire
   - Pin 3 to pin 4 - ``TM-`` to ``LP-``, Blue wire to Grey wire

   The first test is a loopback test which uses the ``TR`` differential signal output to test the
   AGC status input. If this test passes you can be confident that the entire path through the
   differential driver and receiver works properly. It will alternate between setting and
   clearing the ``TR`` signal.

   - Verify the hex digit printed by the script is ``0x20`` when the output pin is high.
   - Verify the hex digit printed by the script is ``0x800`` when the output pin is low.
   - If you see ``0xa20`` or ``0xa00`` during this test, verify the loop-back connections are in place.
   - Proceed to the next test (CTRL+C, then enter "y").

#. The second test is a loopback test which uses the ``TM`` differential signal output to test the
   Low Power (LP) status input. If this test passes you can be confident that the entire path
   through the differential driver and receiver works properly. It will alternate between
   setting and clearning the ``TM`` signal.

   - Verify the hex digit printed by the script is ``0x2000`` when the output pin is high.
   - Verify the hex digit printed by the script is ``0x200`` when the output pin is low.
   - If you see ``0x2a00`` or ``0xa00`` during this test, verify the loop-back connections are in place.
   - Press CTRL+C, then enter "y" to end the tests.

This concludes the tests! If any of these signal output tests failed, additional troubleshooting
is needed. To check the entire logic path of each signal, follow the testing procedures found in
the TXIO notes document.

Finally, install the enclosure cover lid back in place, ensuring that no wires are pinched.

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
