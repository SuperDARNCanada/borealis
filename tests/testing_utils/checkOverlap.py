#!/usr/bin/env python


def checkOverlap(array):
    list_of_overlaps = []
    for pulse_index in range(0, len(array)):
        pulse_overlap = []
        for next_pulse_index in range(0, len(array)):
            if (
                array[pulse_index] != array[next_pulse_index]
                and array[pulse_index][1] < array[next_pulse_index][2]
                and array[next_pulse_index][1] < array[pulse_index][2]
            ):
                pulse_overlap.append(
                    (array[pulse_index][1], array[next_pulse_index][2])
                )
        list_of_overlaps.append(pulse_overlap)
    return list_of_overlaps


array = [
    ["a", 0, 300],
    ["b", 0, 150],
    ["c", 1500, 1800],
    ["d", 1500, 1650],
    ["e", 1800, 2100],
    ["f", 2400, 2700],
    ["g", 15000, 15300],
    ["h", 15000, 15700],
    ["i", 19000, 19150],
]

print(array)
overlaps = checkOverlap(array)

# Now use a generator to get minimum values and maximum values for the pulse overlaps
# see http://stackoverflow.com/questions/16036913/minimum-of-list-of-lists
