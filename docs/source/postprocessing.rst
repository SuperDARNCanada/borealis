==============
Postprocessing
==============
An improvement of the Borealis system is the storage of lower-level data products, specifically antenna-level I&Q
samples. The generation of these files allows higher-level data products such as RAWACF files to be re-processed,
for example, to fix a mistake in the original processing or to beamform in new directions after the data has already
been collected. A Python package,
`borealis_postprocessors <https://github.com/SuperDARNCanada/borealis_postprocessors>`_, has been written for exactly
this purpose. The package contains an identical processing chain to Borealis, as well as additional processing classes
for novel post-processing. It is written in an easily extensible way to facilitate the development of new processing
capabilities.

As an example of the usefulness of this package, we have used it to implement bistatic experiments using pairs of
SuperDARN radars. In order to analyze bistatic data received by a radar, we wanted to first ensure that the transmitting
radar in the pair was indeed operating for a given time. To do so, we extended ``borealis_postprocessors`` to extract
the timestamps from each data file of the transmitting radar during the bistatic experiment, then transferred these
timestamp files to the receiving radar computer. Next, we created another class in ``borealis_postprocessors`` to
read in the timestamp files and only post-process data with matching timestamps from the receiving radar files. This
would have been impossible to do in real-time given the limited bandwidth and internet speeds of the radar sites.

For further information on ``borealis_postprocessors``, check out the GitHub page
`here <https://github.com/SuperDARNCanada/borealis_postprocessors>`_. Any comments, suggestions, or contributions to
the package are encouraged!
