"""
This file contains class definition for elements in 13 Tangram.
Actually there are not clear (we don't need much) information for
    each different element.
Just store some coordinates is enough.
"""

import numpy as np

from thirteen_tangram.config import ANGEL_NUM, ROTATION_MATRIX, STANDARD_COORDINATES


class Element:

    def __init__(self, element_id):
        self.element_id = element_id
        self.base_coordinates = STANDARD_COORDINATES[element_id].astype(np.int32)  # [k, 2]
        self.coordinates = None  # determined by rotation and base_p
        self.area = len(self.base_coordinates)

    def set_points(self, p, position):
        """Rotate the element.
        Input:
            p: np.array([x, y])
        """
        assert position in range(ANGEL_NUM)
        rotation_matrix = ROTATION_MATRIX[position].astype(np.int32)
        coordinates = np.dot(self.base_coordinates, rotation_matrix).astype(np.int32)
        self.coordinates = (coordinates + p[None, :]).astype(np.int32)

    def display(self):
        """For debug."""
        for i in range(self.area):
            print('p', i, self.coordinates[i])
