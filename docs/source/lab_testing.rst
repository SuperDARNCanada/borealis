===========
Lab Testing
===========

Good lab tests to include before deployement include:

1. Loopback tests at boresight
   - allows you to see the differences in channel power after digitizing
   - verifies that N200's are synchronized, this is due to boresite steering having no phase offset so with all equal cable lengths the phase output on all antennas should be the same (check with rawrf and antennas_iq)
   - scripts are available under tools/testing_utils/plot_borealis_hdf5_data/

2. Logic analyzer tests with a transmitter

3. Long-term reliability tests of software

4. Scope output tests
   - verify pulse shape of TX out
   - verify GPIO signals (T/R)
   - verify pulse distances

5. Test for GPS lock on GPS Octoclock

6. 

