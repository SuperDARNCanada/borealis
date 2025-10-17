.. _realtime-package:

================
Realtime Package
================

The realtime package contains the utilities that check data files for corruption during operation
and serving the data to external sockets.

Data is checked using `pyDARNio <https://github.com/SuperDARN/pyDARNio>`_. Additional data processing is done using a
Python package called `backscatter <https://github.com/SuperDARNCanada/backscatter>`_, which is an implementation of the
standard FITACF3 algorithm for analyzing SuperDARN data.

This module is standalone and thus not needed to operate a Borealis system. If you do wish to use it, ensure that
the ``realtime_address`` field in your config file (see :ref:`config-options`) is set to a device and port that are
properly configured. You can test this by running ``ip addr`` in a terminal and choosing an internet-connected
device that is ``UP``. For example, if you configure ``realtime_address : "tcp://eth1:9696"``, then running ``ip addr``
should see ``UP`` in the ``eth1`` line like so:

.. code-block:: text

    3: eth1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 04:42:1a:ea:a2:c2 brd ff:ff:ff:ff:ff:ff
    altname enp115s0
    inet xx.xx.xx.xx/23 brd xx.xx.xx.xx scope global dynamic noprefixroute eth1
       valid_lft 59988sec preferred_lft 59988sec
    inet6 xxxx:xxxx:xxxx:xxxx:xxxx:xxxx/64 scope link noprefixroute
       valid_lft forever preferred_lft forever

To disable this module, run the radar with the argument ``--realtime-off``. For example::

    /home/radar/borealis/scripts/steamed_hams.py twofsound release --realtime-off
