"""Config file for any input Tangram puzzle."""

import math

# should always use all 7 elements
ELEMENT_NUM = 7
TANGRAM_S = 8.

# I use search per pixel to solve this puzzle (since corners can't guide our search in this case)
# in order to reduce search space and complexity, I resize the input image to a certain unit_length
# also, I think it's impossible to overlap all the area of the input image
# so we just want to overlap a ratio of it
UNIT_LENGTH = 16.

# expected ratio to be filled with Tangram elements
EXPECTED_RATIO = 0.8

# since there may actually be many possible solutions for an input image
# we don't just return the first solution
# but return the one of the first MAX_NUM solutions that has the max IoU with the input image
MAX_NUM = 10

PI = math.pi
SQRT_2 = 2 ** 0.5

# number of angles tried to place elements
# for example, 8 means every rotation is 2 * pi / 8 = pi / 4
ANGLE_NUM = 8  # I've tried 16 but the result is not better than 8...

# not iterate all pixels but every pre-defined intervals!
# use 0.5 * unit_length to reduce searching time!
SLIDING_INTERVAL = int(UNIT_LENGTH * 0.5)

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
