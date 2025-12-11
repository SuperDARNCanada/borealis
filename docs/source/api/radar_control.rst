=============
Radar Control
=============

The radar_control package contains a single module, radar_control, that is a
standalone program.

Usage
-----

.. argparse::
   :module: src.radar_control
   :func: radctrl_parser
   :prog: radar_control.py

API
---

.. automodule:: src.radar_control
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: RadctrlParameters, CFSParameters, radctrl_parser

.. autoclass:: src.radar_control.CFSParameters
   :class-doc-from: class
   :members:
   :exclude-members: __init__

.. autoclass:: src.radar_control.RadctrlParameters
   :class-doc-from: class
   :members:
   :exclude-members: __init__
