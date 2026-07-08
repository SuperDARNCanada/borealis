import os
import sys
import unittest

from pydantic import ValidationError

# Need the path append to import within this file
BOREALISPATH = os.environ["BOREALISPATH"]
sys.path.append(f"{BOREALISPATH}/src")

from utils.frequencies import (
    Band,
    TuneBand,
    create_bands,
    find_overlap,
    determine_tuning_freqs,
    snap_to_usrp_clocks,
)


class TestBand(unittest.TestCase):
    def test_init(self):
        # happy cases, should all work
        Band(8000, 20000)
        Band(8000.0, 20000)
        Band(8000, 20000.0)
        Band(8000.0, 20000.0)
        Band(-9000, -8000)

        self.assertRaises(ValidationError, Band, 20000, 8000)
        self.assertRaises(ValidationError, Band, 8000, 8000)
        self.assertRaises(ValidationError, Band, -8000, -9000)

    def test_contains_freq(self):
        band = Band(8000, 20000)
        self.assertTrue(band.contains_freq(10000))
        self.assertTrue(band.contains_freq(8000))
        self.assertTrue(band.contains_freq(8000.0))
        self.assertTrue(band.contains_freq(20000))
        self.assertTrue(band.contains_freq(20000.0))
        self.assertFalse(band.contains_freq(7999.99))
        self.assertFalse(band.contains_freq(20000.0000001))
        with self.assertRaises(ValidationError):
            band.contains_freq("8001")

    def test_contains_all_freqs(self):
        band = Band(8000, 20000)
        self.assertTrue(band.contains_all_freqs([10000, 8000, 8000.0, 20000, 20000.0]))
        self.assertFalse(band.contains_all_freqs([10000, 7999.99, 9590]))
        self.assertFalse(band.contains_all_freqs([12000, 8150, 20000.0000001]))

    def test_contains_freq_or_range(self):
        band = Band(8000, 20000)
        # freq checks
        self.assertTrue(band.contains_freq_or_range(10000))
        self.assertTrue(band.contains_freq_or_range(8000))
        self.assertTrue(band.contains_freq_or_range(8000.0))
        self.assertTrue(band.contains_freq_or_range(20000))
        self.assertTrue(band.contains_freq_or_range(20000.0))
        self.assertFalse(band.contains_freq_or_range(7999.99))
        self.assertFalse(band.contains_freq_or_range(20000.0000001))

        # range checks
        self.assertTrue(band.contains_freq_or_range(Band(9000, 9500)))
        self.assertTrue(band.contains_freq_or_range(Band(8000, 8500)))
        self.assertTrue(band.contains_freq_or_range(Band(19500, 20000)))
        self.assertFalse(band.contains_freq_or_range(Band(7999, 8500)))
        self.assertFalse(band.contains_freq_or_range(Band(19500, 20000.0001)))
        self.assertFalse(band.contains_freq_or_range(Band(7000, 7500)))
        self.assertFalse(band.contains_freq_or_range(Band(-10000, -9000)))

    def test_shift_down(self):
        band = Band(8000, 20000)
        new_band = band.shift_down(100)
        self.assertEqual(new_band.min, 7900)
        self.assertEqual(new_band.max, 19900)
        new_band = band.shift_down(-100)
        self.assertEqual(new_band.min, 8100)
        self.assertEqual(new_band.max, 20100)
        with self.assertRaises(ValidationError):
            band.shift_down("100")


class TestTuneBand(unittest.TestCase):
    def test_init(self):
        # happy tests
        low_band = Band(8000, 9000)
        high_band = Band(10000, 11000)
        TuneBand(low_band, high_band)

        with self.assertRaises(ValidationError):
            TuneBand(high_band, low_band)
        with self.assertRaises(ValidationError):
            TuneBand(low_band, Band(9000, 9100))

    def test_contains_freq(self):
        low_band = Band(8000, 9000)
        high_band = Band(10000, 11000)
        tb = TuneBand(low_band, high_band)

        self.assertTrue(tb.contains_freq(8000))
        self.assertTrue(tb.contains_freq(8500))
        self.assertTrue(tb.contains_freq(9000))
        self.assertTrue(tb.contains_freq(10000))
        self.assertTrue(tb.contains_freq(10500))
        self.assertTrue(tb.contains_freq(11000))
        self.assertFalse(tb.contains_freq(7999))
        self.assertFalse(tb.contains_freq(9000.001))
        self.assertFalse(tb.contains_freq(9999.999))
        self.assertFalse(tb.contains_freq(11000.0001))

    def test_contains_freq_or_range(self):
        low_band = Band(8000, 9000)
        high_band = Band(10000, 11000)
        tb = TuneBand(low_band, high_band)

        # single frequencies
        self.assertTrue(tb.contains_freq(8000))
        self.assertTrue(tb.contains_freq(8500))
        self.assertTrue(tb.contains_freq(9000))
        self.assertTrue(tb.contains_freq(10000))
        self.assertTrue(tb.contains_freq(10500))
        self.assertTrue(tb.contains_freq(11000))
        self.assertFalse(tb.contains_freq(7999))
        self.assertFalse(tb.contains_freq(9000.001))
        self.assertFalse(tb.contains_freq(9999.999))
        self.assertFalse(tb.contains_freq(11000.0001))

        # frequency ranges
        self.assertTrue(tb.contains_freq_or_range(Band(8000, 8100)))
        self.assertTrue(tb.contains_freq_or_range(Band(8000, 9000)))
        self.assertTrue(tb.contains_freq_or_range(Band(10500, 10800)))
        self.assertTrue(tb.contains_freq_or_range(Band(10500, 11000)))
        self.assertTrue(tb.contains_freq_or_range(Band(10000, 11000)))
        self.assertFalse(tb.contains_freq_or_range(Band(7900, 8100)))
        self.assertFalse(tb.contains_freq_or_range(Band(8900, 9100)))
        self.assertFalse(tb.contains_freq_or_range(Band(9900, 10000)))
        self.assertFalse(tb.contains_freq_or_range(Band(10999, 11001)))
        self.assertFalse(tb.contains_freq_or_range(Band(7999, 9001)))
        self.assertFalse(tb.contains_freq_or_range(Band(9999, 11001)))
        self.assertFalse(tb.contains_freq_or_range(Band(8999, 9001)))

    def test_shift_down(self):
        low_band = Band(8000, 9000)
        high_band = Band(10000, 11000)
        tb = TuneBand(low_band, high_band)

        new_tune = tb.shift_down(100)
        self.assertEqual(new_tune.lower.min, 7900)
        self.assertEqual(new_tune.lower.max, 8900)
        self.assertEqual(new_tune.upper.min, 9900)
        self.assertEqual(new_tune.upper.max, 10900)

        new_tune = tb.shift_down(-100)
        self.assertEqual(new_tune.lower.min, 8100)
        self.assertEqual(new_tune.lower.max, 9100)
        self.assertEqual(new_tune.upper.min, 10100)
        self.assertEqual(new_tune.upper.max, 11100)
        with self.assertRaises(ValidationError):
            tb.shift_down("100")

    def test_determine_shift_bands_for_freq(self):
        low_band = Band(8000, 9000)
        high_band = Band(10000, 11000)
        tb = TuneBand(low_band, high_band)

        shift_bands = tb._determine_shift_bands_for_freq(8500)
        self.assertEqual(len(shift_bands), 2)
        self.assertEqual(shift_bands[0].min, 0)
        self.assertEqual(shift_bands[0].max, 500)
        self.assertEqual(shift_bands[1].min, 1500)
        self.assertEqual(shift_bands[1].max, 2500)

        shift_bands = tb._determine_shift_bands_for_freq(10100)
        self.assertEqual(len(shift_bands), 1)
        self.assertEqual(shift_bands[0].min, 0)
        self.assertEqual(shift_bands[0].max, 900)

        shift_bands = tb._determine_shift_bands_for_freq(9000)
        self.assertEqual(len(shift_bands), 1)
        self.assertEqual(shift_bands[0].min, 1000)
        self.assertEqual(shift_bands[0].max, 2000)

        shift_bands = tb._determine_shift_bands_for_freq(11000)
        self.assertEqual(len(shift_bands), 0)

        with self.assertRaises(ValueError):
            tb._determine_shift_bands_for_freq(7900)
        with self.assertRaises(ValueError):
            tb._determine_shift_bands_for_freq(9001)
        with self.assertRaises(ValueError):
            tb._determine_shift_bands_for_freq(9900)
        with self.assertRaises(ValueError):
            tb._determine_shift_bands_for_freq(11001)

    def test_determine_shift_bands_for_range(self):
        low_band = Band(8000, 9000)
        high_band = Band(10000, 11000)
        tb = TuneBand(low_band, high_band)

        shift_bands = tb._determine_shift_bands_for_range(Band(8500, 8800))
        self.assertEqual(len(shift_bands), 2)
        self.assertEqual(shift_bands[0].min, 0)
        self.assertEqual(shift_bands[0].max, 200)
        self.assertEqual(shift_bands[1].min, 1500)
        self.assertEqual(shift_bands[1].max, 2200)

        shift_bands = tb._determine_shift_bands_for_range(Band(10100, 10150))
        self.assertEqual(len(shift_bands), 1)
        self.assertEqual(shift_bands[0].min, 0)
        self.assertEqual(shift_bands[0].max, 850)

        shift_bands = tb._determine_shift_bands_for_range(Band(10100, 11000))
        self.assertEqual(len(shift_bands), 0)

        shift_bands = tb._determine_shift_bands_for_range(Band(8900, 9000))
        self.assertEqual(len(shift_bands), 1)
        self.assertEqual(shift_bands[0].min, 1100)
        self.assertEqual(shift_bands[0].max, 2000)

        with self.assertRaises(ValueError):
            tb._determine_shift_bands_for_range(Band(7900, 8200))
        with self.assertRaises(ValueError):
            tb._determine_shift_bands_for_range(Band(8800, 9001))
        with self.assertRaises(ValueError):
            tb._determine_shift_bands_for_range(Band(9900, 10100))
        with self.assertRaises(ValueError):
            tb._determine_shift_bands_for_range(Band(10800, 11001))

    def test_allowed_shifts(self):
        low_band = Band(8000, 9000)
        high_band = Band(10000, 11000)
        tb = TuneBand(low_band, high_band)

        # Single frequency tests
        shift_bands = tb._determine_shift_bands_for_freq(8500)
        self.assertEqual(len(shift_bands), 2)
        self.assertEqual(shift_bands[0].min, 0)
        self.assertEqual(shift_bands[0].max, 500)
        self.assertEqual(shift_bands[1].min, 1500)
        self.assertEqual(shift_bands[1].max, 2500)

        shift_bands = tb._determine_shift_bands_for_freq(10100)
        self.assertEqual(len(shift_bands), 1)
        self.assertEqual(shift_bands[0].min, 0)
        self.assertEqual(shift_bands[0].max, 900)

        shift_bands = tb._determine_shift_bands_for_freq(9000)
        self.assertEqual(len(shift_bands), 1)
        self.assertEqual(shift_bands[0].min, 1000)
        self.assertEqual(shift_bands[0].max, 2000)

        shift_bands = tb._determine_shift_bands_for_freq(11000)
        self.assertEqual(len(shift_bands), 0)

        with self.assertRaises(ValueError):
            tb._determine_shift_bands_for_freq(7900)
        with self.assertRaises(ValueError):
            tb._determine_shift_bands_for_freq(9001)
        with self.assertRaises(ValueError):
            tb._determine_shift_bands_for_freq(9900)
        with self.assertRaises(ValueError):
            tb._determine_shift_bands_for_freq(11001)

        # Frequency range tests
        shift_bands = tb._determine_shift_bands_for_range(Band(8500, 8800))
        self.assertEqual(len(shift_bands), 2)
        self.assertEqual(shift_bands[0].min, 0)
        self.assertEqual(shift_bands[0].max, 200)
        self.assertEqual(shift_bands[1].min, 1500)
        self.assertEqual(shift_bands[1].max, 2200)

        shift_bands = tb._determine_shift_bands_for_range(Band(10100, 10150))
        self.assertEqual(len(shift_bands), 1)
        self.assertEqual(shift_bands[0].min, 0)
        self.assertEqual(shift_bands[0].max, 850)

        shift_bands = tb._determine_shift_bands_for_range(Band(10100, 11000))
        self.assertEqual(len(shift_bands), 0)

        shift_bands = tb._determine_shift_bands_for_range(Band(8900, 9000))
        self.assertEqual(len(shift_bands), 1)
        self.assertEqual(shift_bands[0].min, 1100)
        self.assertEqual(shift_bands[0].max, 2000)

        with self.assertRaises(ValueError):
            tb._determine_shift_bands_for_range(Band(7900, 8200))
        with self.assertRaises(ValueError):
            tb._determine_shift_bands_for_range(Band(8800, 9001))
        with self.assertRaises(ValueError):
            tb._determine_shift_bands_for_range(Band(9900, 10100))
        with self.assertRaises(ValueError):
            tb._determine_shift_bands_for_range(Band(10800, 11001))

    def test_shift_to_accommodate(self):
        low_band = Band(8000, 9000)
        high_band = Band(10000, 11000)
        tb = TuneBand(low_band, high_band)

        # Single frequency value
        shift_bands = tb.shift_to_accommodate(7500)
        self.assertEqual(len(shift_bands), 2)
        self.assertEqual(shift_bands[0].min, 500)
        self.assertEqual(shift_bands[0].max, 1500)
        self.assertEqual(shift_bands[1].min, 2500)
        self.assertEqual(shift_bands[1].max, 3500)

        shift_bands = tb.shift_to_accommodate(9000)
        self.assertEqual(len(shift_bands), 1)
        self.assertEqual(shift_bands[0].min, 1000)
        self.assertEqual(shift_bands[0].max, 2000)

        shift_bands = tb.shift_to_accommodate(9800)
        self.assertEqual(len(shift_bands), 1)
        self.assertEqual(shift_bands[0].min, 200)
        self.assertEqual(shift_bands[0].max, 1200)

        shift_bands = tb.shift_to_accommodate(11004)
        self.assertEqual(len(shift_bands), 0)

        # Frequency range inputs
        shift_bands = tb.shift_to_accommodate(Band(7500, 7800))
        self.assertEqual(len(shift_bands), 2)
        self.assertEqual(shift_bands[0].min, 500)
        self.assertEqual(shift_bands[0].max, 1200)
        self.assertEqual(shift_bands[1].min, 2500)
        self.assertEqual(shift_bands[1].max, 3200)

        shift_bands = tb.shift_to_accommodate(Band(8900, 9100))
        self.assertEqual(len(shift_bands), 1)
        self.assertEqual(shift_bands[0].min, 1100)
        self.assertEqual(shift_bands[0].max, 1900)

        shift_bands = tb.shift_to_accommodate(Band(9900, 10100))
        self.assertEqual(len(shift_bands), 1)
        self.assertEqual(shift_bands[0].min, 100)
        self.assertEqual(shift_bands[0].max, 900)

        shift_bands = tb.shift_to_accommodate(Band(10900, 11000))
        self.assertEqual(len(shift_bands), 0)


class TestCreateBands(unittest.TestCase):
    def test_call(self):
        tb = create_bands(12000, 5e6)
        self.assertEqual(tb.lower.min, 11999)
        self.assertEqual(tb.lower.max, 13724)
        self.assertEqual(tb.upper.min, 13774)
        self.assertEqual(tb.upper.max, 15499)

        tb = create_bands(Band(12000, 12300), 5e6)
        self.assertEqual(tb.lower.min, 11999)
        self.assertEqual(tb.lower.max, 13724)
        self.assertEqual(tb.upper.min, 13774)
        self.assertEqual(tb.upper.max, 15499)

        with self.assertRaises(ValueError):
            tb = create_bands(Band(12000, 14000), 5e6)


class TestFindOverlap(unittest.TestCase):
    def test_call(self):
        current_bands = [
            Band(8000, 9000),
            Band(9500, 10000),
            Band(11000, 12000),
            Band(12500, 13500),
        ]
        new_bands = [Band(8500, 9200), Band(10800, 12800), Band(14000, 14500)]
        overlap = find_overlap(current_bands, new_bands)
        self.assertEqual(len(overlap), 3)
        self.assertEqual(overlap[0].min, 8500)
        self.assertEqual(overlap[0].max, 9000)
        self.assertEqual(overlap[1].min, 11000)
        self.assertEqual(overlap[1].max, 12000)
        self.assertEqual(overlap[2].min, 12500)
        self.assertEqual(overlap[2].max, 12800)


class TestDetermineTuningFreqs(unittest.TestCase):
    def test_call(self):
        sampling_rate = 5e6
        master_clock_rate = 100e6

        # can be tuned together
        freqs = [10001, 13000]
        expected_tunes = [11750]
        tuning_freqs, groups = determine_tuning_freqs(
            freqs, sampling_rate, master_clock_rate
        )
        self.assertEqual(groups, [[0, 1]])
        for actual, expected in zip(tuning_freqs, expected_tunes):
            self.assertEqual(actual, snap_to_usrp_clocks(expected, master_clock_rate))

        # can be tuned together, but needs adjusting
        freqs = [10000, 9000]
        expected_tunes = [10750]
        tuning_freqs, groups = determine_tuning_freqs(
            freqs, sampling_rate, master_clock_rate
        )
        self.assertEqual(groups, [[0, 1]])
        for actual, expected in zip(tuning_freqs, expected_tunes):
            self.assertEqual(actual, snap_to_usrp_clocks(expected, master_clock_rate))

        # 2nd frequency in center restricted band, tune adjusted to be lower
        freqs = [10001, 11750]
        expected_tunes = [11725]
        tuning_freqs, groups = determine_tuning_freqs(
            freqs, sampling_rate, master_clock_rate
        )
        self.assertEqual(groups, [[0, 1]])
        for actual, expected in zip(tuning_freqs, expected_tunes):
            self.assertEqual(actual, snap_to_usrp_clocks(expected, master_clock_rate))

        # last frequency in center restricted band, need two groups
        freqs = [10001, 13500, 11750]
        expected_tunes = [11750, 13499]
        tuning_freqs, groups = determine_tuning_freqs(
            freqs, sampling_rate, master_clock_rate
        )
        self.assertEqual(groups, [[0, 1], [2]])
        for actual, expected in zip(tuning_freqs, expected_tunes):
            self.assertEqual(actual, snap_to_usrp_clocks(expected, master_clock_rate))

        # larger example, from SAS sounding frequencies
        freqs = [9690, 10440, 11500, 12080, 13000, 14560, 15250, 16400]
        expected_tunes = [11439, 16309]
        expected_groups = [[0, 1, 2, 3, 4], [5, 6, 7]]
        tuning_freqs, groups = determine_tuning_freqs(
            freqs, sampling_rate, master_clock_rate
        )
        self.assertEqual(groups, expected_groups)
        for actual, expected in zip(tuning_freqs, expected_tunes):
            self.assertEqual(actual, snap_to_usrp_clocks(expected, master_clock_rate))


if __name__ == "__main__":
    unittest.main()
