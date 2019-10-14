"""Config file for thirteen elements Tangram puzzle."""

import math
import numpy as np

# in 13 Tangram puzzle we don't always use all 13 elements! (actually only 9 or 10 most of the times)
ELEMENT_NUM = 13
TANGRAM_S = 64  # 12 * 5 + 4
PI = math.pi

# number of angles tried
ANGEL_NUM = 4  # only need to try 4 (0, 90, 180, 270) rotations
ROTATION_MATRIX = [
    np.array([[1, 0], [0, 1]]),
    np.array([[0, -1], [1, 0]]),
    np.array([[-1, 0], [0, -1]]),
    np.array([[0, 1], [-1, 0]])
]

# standard coordinates of elements, square is the first one
STANDARD_COORDINATES = {
    '0': np.array([(0, 0), (1, 0), (0, 1), (1, 1)]),  # square
    '1': np.array([(0, 0), (1, 0), (1, -1), (1, -2), (2, -2)]),
    '2': np.array([(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)]),
    '3': np.array([(0, 0), (0, -1), (0, -2), (1, -2), (1, -3)]),
    '4': np.array([(0, 0), (1, 0), (1, 1), (1, -1), (2, 0)]),
    '5': np.array([(0, 0), (1, 0), (1, 1), (2, 0), (2, -1)]),
    '6': np.array([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]),
    '7': np.array([(0, 0), (0, 1), (0, 2), (0, 3), (1, 2)]),
    '8': np.array([(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)]),
    '9': np.array([(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]),
    '10': np.array([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]),
    '11': np.array([(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)]),
    '12': np.array([(0, 0), (1, 0), (0, 1), (1, 1), (1, 2)])
}

# colors for drawing elements of Tangram
COLORS = [
    (int(255. * i / ELEMENT_NUM),
     int(128. + 255. * i / ELEMENT_NUM) % 255,
     int(255. - 255. * i / ELEMENT_NUM))
    for i in range(ELEMENT_NUM)
]
