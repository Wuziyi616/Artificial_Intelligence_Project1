"""Config file for 7 elements Tangram puzzle."""

import math

# in 7 Tangram we always use all the elements
ELEMENT_NUM = 7
TANGRAM_S = 8.
PI = math.pi
SQRT_2 = 2 ** 0.5

# number of angles tried to place elements
# for example, 8 means every rotation is 2 * pi / 8 = pi / 4
ANGLE_NUM = 8

# standard size of elements in a set of Tangram, the first element is its area
STANDARD_SIZE = {
    'large_triangle1': [2., 2 * SQRT_2, 2., 2.],
    'large_triangle2': [2., 2 * SQRT_2, 2., 2.],
    'medium_triangle': [1., 2., SQRT_2, SQRT_2],
    'small_triangle1': [0.5, SQRT_2, 1., 1.],
    'small_triangle2': [0.5, SQRT_2, 1., 1.],
    'square': [1., 1., 1., 1., 1.],
    'parallelogram': [1., SQRT_2, 1., SQRT_2, 1.]
}

# reference for element id and element name, following the order to place them
ELEMENT_DICT = {
    '1': 'large_triangle1',
    '2': 'large_triangle2',
    '3': 'square',
    '4': 'parallelogram',
    '5': 'medium_triangle',
    '6': 'small_triangle1',
    '7': 'small_triangle2'
}

# total state number for each element
TRIANGLE_STATE = 3  # p0, p1, p2
SQUARE_STATE = 1  # p0
PARALLELOGRAM_STATE = 4  # p0, p1, inverse_p0, inverse_p1

# colors for drawing elements of Tangram
COLORS = [
    (int(255. * i / ELEMENT_NUM),
     int(128. + 255. * i / ELEMENT_NUM) % 255,
     int(255. - 255. * i / ELEMENT_NUM))
    for i in range(ELEMENT_NUM)
]
