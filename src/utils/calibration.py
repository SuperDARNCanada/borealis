import numpy as np


class Calibration:
    """
    Class for holding calibration values. Expects a dictionary with keys "cable_velocity", "channels", where
    "channels" is a dictionary keyed by integer channel numbers.
    """

    def __init__(self, d: dict):
        self.cable_velocity: float = d["cable_velocity"]
        cals = dict()
        for k, v in d["channel"].items():
            cals[int(k)] = v
        setattr(self, "channels", cals)


def cable_len_to_elec_len(
    cable_len: float, vel_factor: float, freq_hz: list[float]
) -> list[float]:
    """
    Converts a cable length in meters to an electrical length in radians.
    """
    elec_len = [
        (f / (299_792_458 * vel_factor)) * cable_len * 2.0 * np.pi for f in freq_hz
    ]

    return elec_len
