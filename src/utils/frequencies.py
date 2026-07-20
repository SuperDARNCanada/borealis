"""
frequencies.py
~~~~~~~~~~~~~~

The N200 devices are operated at a given sampling rate, and operable frequencies are specified by tuning the N200 to
mix the operating band up such that it is centered around the tuning frequency. However, not all parts of the band are
usable due to distortions around the tuning frequency and from the intrinsic N200 anti-aliasing filter. As such,
determination of tuning frequencies is somewhat complex as there are effectively 5 regions to consider:

    |xxxxxx------------xxxxx------------xxxxxx|
    |  1  |      2     | 3 |     4      |  5  |

where the total width (between the outer pipes) is the sampling rate.
Regions 1 and 5 are each (sampling_rate * 0.15) wide, and unusable due to anti-aliasing filter distortion.
Region 3 is 50 kHz wide, and unusable due to distortion around the tuning frequency. This was determined empirically.
Regions 2 and 4 are each ((sampling_rate / 2) - 25 kHz) wide, covering the remainder of the band.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import List, Tuple, Union

from pydantic import validate_call, ConfigDict, model_validator
from pydantic.dataclasses import dataclass

F = Union[int, float]
"""Alias for Union[int, float]"""

FreqOrRange = Union[F, "Band"]
"""Alias that covers a frequency specification that is either a number or a range of frequencies"""


validation_config = ConfigDict(
    validate_assignment=True,
    validate_default=True,
    extra="forbid",
    arbitrary_types_allowed=False,
    strict=True,
)


@dataclass(config=validation_config)
class Band:
    """Frequency band"""

    min: F
    max: F

    @model_validator(mode="after")
    def check_bounds(self):
        if self.max <= self.min:
            raise ValueError(f"Band is invalid, min >= max ({self.min} >= {self.max}).")

        return self

    @validate_call(config=validation_config)
    def contains_freq(self, freq: F) -> bool:
        """
        Returns True if `freq` lies within `self`.
        """
        return self.min <= freq <= self.max

    @validate_call(config=validation_config)
    def contains_all_freqs(self, freqs: Iterable[F]) -> bool:
        """
        Returns True if all frequencies lie in the band.
        """
        return all([self.contains_freq(f) for f in freqs])

    def contains_freq_or_range(self, freq: Union[F, "Band"]) -> bool:
        """
        Returns True if the `freq` lies completely within the band.
        """
        if isinstance(freq, (int, float)):
            return self.contains_freq(freq)
        elif isinstance(freq, Band):
            return self.contains_all_freqs([freq.min, freq.max])
        else:
            raise TypeError(
                f"Expected one of (int, float, Band), found {type(freq)} instead"
            )

    @validate_call(config=validation_config)
    def shift_down(self, amount: F):
        """
        Shifts the band down by `amount`
        """
        return Band(self.min - amount, self.max - amount)


@dataclass(config=validation_config)
class TuneBand:
    """Two frequencies bands defining the operable frequencies given a tuning frequency."""

    lower: Band
    upper: Band

    @model_validator(mode="after")
    def check_bands(self):
        if self.lower.max >= self.upper.min:
            raise ValueError(
                f"Band is invalid, lower.max >= upper.min ({self.lower.max} >= {self.upper.min})."
            )

        return self

    @validate_call(config=validation_config)
    def contains_freq(self, freq: F) -> bool:
        """
        Returns True if freq lies in one of the bands.
        """
        return any([b.contains_freq(freq) for b in [self.lower, self.upper]])

    @validate_call(config=validation_config)
    def contains_freq_or_range(self, x: "FreqOrRange") -> bool:
        """
        Returns True if the input lies entirely within one of the bands of `self`.
        """
        return any([b.contains_freq_or_range(x) for b in [self.lower, self.upper]])

    @validate_call(config=validation_config)
    def _determine_shift_bands_for_freq(self, freq: F) -> List[Band]:
        """
        Determines ranges of how much `self` can be shifted down while still containing `freq`.

        There are two cases: `freq` is contained in `self.lower`, or `freq` is contained in `self.upper`. The first
        case is illustrated below.

        regions:                                           1          2        3         4        5
        self:                                           |xxxxxx|------------|xxxxx|------------|xxxxxx|
        freq:                                                         O
        upper edge of shift 1:                     |xxxxxx|-----------O|xxxxx|------------|xxxxxx| <--
        lower edge of shift 2:             |xxxxxx|------------|xxxxx|O-----------|xxxxxx| <----------
        upper edge of shift 2:  |xxxxxx|------------|xxxxx|-----------O|xxxxxx| <---------------------

        There are two ranges of allowed shift for this case:
           1. shift `self` down until `freq` hits Region 3 (`self.lower.max`)
           2. shift `self` such that `freq` is contained in Region 4 (`self.upper`)

        For the case where `band` starts out in Region 4 (contained in `self.upper`), it is simpler.

        regions:                                           1          2        3         4        5
        self:                                           |xxxxxx|------------|xxxxx|------------|xxxxxx|
        band:                                                                         O
        edge of shift 1:                         |xxxxxx|------------|xxxxx|----------O|xxxxxx| <----
        """
        shift_bands = []
        if self.lower.contains_freq(freq):
            if self.lower.max != freq:
                shift_bands.append(Band(0, self.lower.max - freq))
            shift_bands.append(Band(self.upper.min - freq, self.upper.max - freq))
        elif self.upper.contains_freq(freq):
            if self.upper.max != freq:
                shift_bands.append(Band(0, self.upper.max - freq))
        else:
            raise ValueError(f"Freq {freq} not in bands {self}")
        return shift_bands

    @validate_call(config=validation_config)
    def _determine_shift_bands_for_range(self, band: Band) -> List[Band]:
        """
        Determines ranges of how much `self` can be shifted down while still containing `band`.

        Now there are two bands, the lower one is as narrow as possible since `band.max` will hit Region 3 soonest.
        The upper band must be shrunk, since at the low end of it when `band.max` just enters Region 4,
        `band.min` is still stuck in Region 3. E.g.

        There are two cases: `band` is contained in `self.lower`, or `band` is contained in `self.upper`. The first
        case is illustrated below.

        regions:                                           1          2        3         4        5
        self:                                           |xxxxxx|------------|xxxxx|------------|xxxxxx|
        band:                                                        OO
        edge of shift 1:                           |xxxxxx|----------OO|xxxxx|------------|xxxxxx| <--
        shift 2:                          |xxxxxx|------------|xxxxx|OO----------|xxxxxx| <-----------
        edge of shift 2:        |xxxxxx|------------|xxxxx|----------OO|xxxxxx| <---------------------

        There are two ranges of allowed shift: the first, we shift self to the left until the right side of `band`
        hits Region 3. The second, after we've popped fully into Region 4, shift until again the right side of `band`
        approaches the edge of Region 4.

        For the case where `band` starts out in Region 4 (contained in `self.upper`), it is simpler.

        regions:                                           1          2        3         4        5
        self:                                           |xxxxxx|------------|xxxxx|------------|xxxxxx|
        band:                                                                         OO
        edge of shift 1:                         |xxxxxx|------------|xxxxx|----------OO|xxxxxx| <----

        There is no second range of allowed shifts.

        """
        if not self.contains_freq_or_range(band):
            raise ValueError(f"Frequency range {band} not contained in bands {self}")

        upper_freq_shift_bands = self._determine_shift_bands_for_freq(band.max)

        if self.upper.contains_freq(band.min):
            return upper_freq_shift_bands
        elif self.lower.max != band.max:
            # nominal case, band has room to shift within self.lower
            # Now there are two bands, the lower one is as narrow as possible since `band.max` will hit Region 3 soonest.
            # The upper band must be shrunk, since at the low end of it when `band.max` just enters Region 4,
            # `band.min` is still stuck in Region 3.
            lower_freq_shift_bands = self._determine_shift_bands_for_freq(band.min)
            upper_band = Band(
                lower_freq_shift_bands[1].min, upper_freq_shift_bands[1].max
            )
            return [upper_freq_shift_bands[0], upper_band]
        else:
            # edge case, there is actually no room to shift in the lower band since the range is already at the edge
            lower_freq_shift_bands = self._determine_shift_bands_for_freq(band.min)
            upper_band = Band(
                lower_freq_shift_bands[1].min, upper_freq_shift_bands[0].max
            )
            return [upper_band]

    @validate_call(config=validation_config)
    def allowed_shifts(self, x: "FreqOrRange") -> List[Band]:
        """
        Determines ranges of how much `self` can be shifted down while still containing `band`.

        There are either one or two ranges of shift, depending on whether the input starts in the lower or upper range
        of `self`.
        """
        if isinstance(x, int) or isinstance(x, float):
            return self._determine_shift_bands_for_freq(x)
        else:
            return self._determine_shift_bands_for_range(x)

    @validate_call(config=validation_config)
    def shift_down(self, amount: F):
        """
        Shifts the bands down by `amount`.
        """
        return TuneBand(self.lower.shift_down(amount), self.upper.shift_down(amount))

    @validate_call(config=validation_config)
    def shift_to_accommodate(self, x: "FreqOrRange") -> List[Band]:
        """
        Determines ranges of how much `self` can be shifted down to encompass the input.

        There are either one or two ranges of shift, depending on whether the input starts below the lower range
        or in the gap between ranges of `self`.
        """
        shift_bands = []
        if isinstance(x, int) or isinstance(x, float):
            if x >= self.upper.max:
                return shift_bands

            if x <= self.lower.min:
                shift_bands.append(Band(self.lower.min - x, self.lower.max - x))
            shift_bands.append(Band(self.upper.min - x, self.upper.max - x))
        else:
            if x.max >= self.upper.max or (
                x.max - x.min > self.lower.max - self.lower.min
            ):
                # either x is above the band, or the range is so great it would never fit
                return shift_bands

            if x.max < self.lower.max:
                shift_bands.append(Band(self.lower.min - x.min, self.lower.max - x.max))
            shift_bands.append(Band(self.upper.min - x.min, self.upper.max - x.max))

        return shift_bands


@validate_call(config=validation_config)
def create_bands(freq: FreqOrRange, sampling_rate_hz: F) -> TuneBand:
    """
    Creates operating bands given a starting frequency or frequency range, slotting the frequency or bottom of the range
    into the very bottom end of Region 2 (see module docstring).
    """
    bandwidth = (0.7 * sampling_rate_hz) / 1000  # kHz
    center_band_to_avoid = (
        50  # kHz, too close to tuning freq and distortion is introduced
    )
    half_bandwidth = (bandwidth - center_band_to_avoid) / 2

    if isinstance(freq, int) or isinstance(freq, float):
        is_band = False
        bottom_of_band = freq - 1  # 1 kHz below freq
    else:
        is_band = True
        bottom_of_band = freq.min - 1  # 1 kHz below bottom of range
    top_of_band = bottom_of_band + bandwidth

    lower_band = Band(bottom_of_band, bottom_of_band + half_bandwidth)
    upper_band = Band(top_of_band - half_bandwidth, top_of_band)

    if is_band and not lower_band.contains_freq_or_range(freq):
        raise ValueError(
            f"Cannot create bands for frequency range, the range {freq} is too large "
            f"given the sampling rate {sampling_rate_hz} Hz"
        )

    return TuneBand(lower_band, upper_band)


@validate_call(config=validation_config)
def find_overlap(current_bands: List[Band], new_bands: List[Band]) -> List[Band]:
    """
    Takes a list of `Band`s that are non-overlapping, and finds the overlap with `new_bands`.

    To illustrate:

        current_bands:    |-------|    |----|       |----------| |-----|
        new_bands:            |-----|            |------------------|     |----|
        result:               |---|                 |----------| |--|

    """
    if any([b.min >= b.max for b in current_bands]):
        raise ValueError(
            f"Bands not internally consistent (max smaller/equal to min): {current_bands}"
        )

    if any(
        [
            current_bands[i - 1].max >= current_bands[i].min
            for i in range(1, len(current_bands))
        ]
    ):
        raise ValueError(
            f"Bands not in order, or they are overlapping: {current_bands}"
        )

    if any([b.min >= b.max for b in new_bands]):
        raise ValueError(
            f"New bands not internally consistent (max smaller/equal to min): {new_bands}"
        )

    if any(
        [new_bands[i - 1].max >= new_bands[i].min for i in range(1, len(new_bands))]
    ):
        raise ValueError(
            f"New bands not in order, or they are overlapping: {new_bands}"
        )

    final = []
    for nb in new_bands:
        for cb in current_bands:
            lower_bound = max(nb.min, cb.min)
            upper_bound = min(nb.max, cb.max)
            if lower_bound < upper_bound:
                final.append(Band(lower_bound, upper_bound))

    return final


@validate_call(config=validation_config)
def find_tune_freq(tune_band: TuneBand) -> float:
    """
    Determines the tuning frequency from `tune_band`
    """
    return (tune_band.lower.max + tune_band.upper.min) / 2.0


@validate_call(config=validation_config)
def determine_tuning_freqs(
    freqs: List[Union[FreqOrRange, List[F]]], sampling_rate: F, master_clock_rate: F
) -> Tuple[List[float], List[List[int]]]:
    """
    Determines the tuning frequencies for each item in `freqs`, attempting to minimize the total number of re-tunes
    between adjacent items in `freqs`.

    This function takes a list of frequencies or frequency ranges (e.g. [10500, 10800, [10900, 11200], 12200])
    and a sampling rate (e.g. 5 MHz) and resolves the tuning frequencies that minimize the number of retuning events
    between adjacent frequencies/ranges. No reordering of frequencies is done, to preserve order-of-operations for
    experiments which use the `freq_order` keyword.

    :param freqs:   Frequencies under consideration. Either a plain frequency in kHz, or a Tuple[int, int] defining
                    a band which must use one tuning frequency (e.g. for clear frequency search).
    :type  freqs:   List[Union[FreqOrRange, List[F]]]
    :param sampling_rate: Sampling rate of the experiment
    :type  sampling_rate: F
    :param master_clock_rate: Clock rate of the USRP devices
    :type  master_clock_rate: F

    :returns:  (tuning freqs, grouped indices) giving the tuning freqs to use, and which indices of `freqs` should
               use each tuning frequency.
    :rtype:    Tuple[List[float], List[List[int]]]
    """
    if len(freqs) == 0:
        return [], []

    for i, f in enumerate(freqs):
        if isinstance(f, list):
            freqs[i] = Band(f[0], f[1])

    tuning_freqs = []
    freq_groups = []
    proposed_bands = create_bands(freqs[0], sampling_rate)

    # Begin calculating the tuning frequencies
    freq_idx = 0
    current_group = [0]

    while freq_idx < len(freqs) - 1:
        next_freq = freqs[freq_idx + 1]
        if proposed_bands.contains_freq_or_range(next_freq):
            # no problem sharing a tuning frequency with this new freq
            current_group.append(freq_idx + 1)
            freq_idx += 1
            continue

        # if we reach here, we've hit a frequency or range which does not lie in the proposed bands

        def start_new_tune_group(c_group, f_idx):
            """Tidy up the tune group and start a new one"""
            tuning_freqs.append(find_tune_freq(proposed_bands))
            freq_groups.append(c_group)

            return [f_idx + 1], f_idx + 1, create_bands(freqs[f_idx + 1], sampling_rate)

        if next_freq > proposed_bands.upper.max:
            # Cannot shift bands up since we always start as high as possible
            current_group, freq_idx, proposed_bands = start_new_tune_group(
                current_group, freq_idx
            )
            continue

        # determine where we can shift the proposed bands before a frequency or range
        # from current_group is removed from the band
        shift_ranges = [Band(0, sampling_rate / 1000)]
        for idx in current_group:
            shifts_for_freq = proposed_bands.allowed_shifts(freqs[idx])
            shift_ranges = find_overlap(shift_ranges, shifts_for_freq)

        shift_for_next_freq = proposed_bands.shift_to_accommodate(next_freq)
        shift_ranges = find_overlap(shift_ranges, shift_for_next_freq)

        if len(shift_ranges) == 0:
            # No shift would accommodate the new frequency, so the current group is complete and we start a new one
            current_group, freq_idx, proposed_bands = start_new_tune_group(
                current_group, freq_idx
            )
            continue

        # If we reach here, shift the proposed bands by the minimum amount to accommodate `next_freq`
        # This leaves the maximum amount of room for any future shifts.
        proposed_bands = proposed_bands.shift_down(shift_ranges[0].min)
        current_group.append(freq_idx + 1)
        freq_idx += 1

    # if we exited the loop above when considering frequencies that all played well
    # (i.e. no frequencies outside `proposed_bands`), then we need to wrap up the last group.
    if len(current_group) > 0:
        tuning_freqs.append(find_tune_freq(proposed_bands))
        freq_groups.append(current_group)

    # Shift the tuning frequencies to align with the USRP clock increments
    tuning_freqs = [snap_to_usrp_clocks(f, master_clock_rate) for f in tuning_freqs]

    return tuning_freqs, freq_groups


@validate_call(config=validation_config)
def snap_to_usrp_clocks(freq: F, master_clock_rate: F):
    """
    Adjust the tuning frequency to correspond to an increment of the USRP clock rate.
    """
    # convert from kHz to Hz to get correct clock divider. Return the result back in kHz.
    clock_multiples = master_clock_rate / 2**32
    clock_divider = math.ceil(freq * 1e3 / clock_multiples)
    ctrfreq = (clock_divider * clock_multiples) / 1e3

    return ctrfreq
