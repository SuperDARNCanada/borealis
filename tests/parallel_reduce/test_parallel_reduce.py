def reduce0(data_list):
    data = data_list[:]
    num_elements = len(data)

    s = 1
    while s < num_elements:
        for i in range(num_elements):
            if i % (2 * s) == 0:
                data[i] += data[i + s]
        s = s * 2

    print("Reduce algorithm 0 result", data[0])


def reduce1(data_list):
    data = data_list[:]
    num_elements = len(data)

    s = 1
    while s < num_elements:
        for i in range(num_elements):
            if i % (2 * s) == 0:
                data[i] += data[i + s]
        s = s * 2

    print("Reduce algorithm 1 result", data[0])


def reduce2(data_list):
    data = data_list[:]
    num_elements = len(data)

    s = num_elements / 2
    while s > 0:
        for i in range(num_elements):
            if i < s:
                data[i] += data[i + s]
        s = s / 2

    print("Reduce algorithm 2 result", data[0])


def reduce3(data_list):
    data = data_list[:]
    num_elements = len(data)

    s = num_elements / 2
    for i in range(s):
        data[i] += data[i + s]

    s = s / 2
    while s > 0:
        for i in range(num_elements):
            if i < s:
                data[i] += data[i + s]
        s = s / 2

    print("Reduce algorithm 3 result", data[0])


def reduce4(data_list):

    data = data_list[:]
    num_elements = len(data)

    s = num_elements / 2
    while s > 32:
        for i in range(num_elements):
            if i < s:
                data[i] += data[i + s]
        s = s / 2

    if num_elements >= 64:
        for i in range(32):
            data[i] += data[i + 32]

    if num_elements >= 32:
        for i in range(16):
            data[i] += data[i + 16]

    if num_elements >= 16:
        for i in range(8):
            data[i] += data[i + 8]

    if num_elements >= 8:
        for i in range(4):
            data[i] += data[i + 4]

    if num_elements >= 4:
        for i in range(2):
            data[i] += data[i + 2]

    if num_elements >= 2:
        for i in range(1):
            data[i] += data[i + 1]

    print("Reduce algorithm 4 result", data[0])


def reduce5(data_list):

    data = data_list[:]
    num_elements = len(data)

    if num_elements >= 512:
        for i in range(256):
            data[i] += data[i + 256]

    if num_elements >= 256:
        for i in range(128):
            data[i] += data[i + 128]

    if num_elements >= 128:
        for i in range(64):
            data[i] += data[i + 64]

    if num_elements >= 64:
        for i in range(32):
            data[i] += data[i + 32]

    if num_elements >= 32:
        for i in range(16):
            data[i] += data[i + 16]

    if num_elements >= 16:
        for i in range(8):
            data[i] += data[i + 8]

    if num_elements >= 8:
        for i in range(4):
            data[i] += data[i + 4]

    if num_elements >= 4:
        for i in range(2):
            data[i] += data[i + 2]

    if num_elements >= 2:
        for i in range(1):
            data[i] += data[i + 1]

    print("Reduce algorithm 5 result", data[0])


data_list = [i for i in range(32)]

reduce0(data_list)
reduce1(data_list)
reduce2(data_list)
reduce3(data_list)
reduce4(data_list)
reduce5(data_list)
