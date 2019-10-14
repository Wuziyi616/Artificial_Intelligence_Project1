"""
This file contains searching algorithm for 13 Tangram task.
The STATE and TRANSFER function is very similar to normal 7 Tangram task.
I will iterate over all pixels in a grid and try placing elements on it.
"""

import copy
import math
import itertools
import numpy as np
import skimage.measure

import thirteen_tangram.image_utils as image_utils
from thirteen_tangram.config import ANGEL_NUM
from thirteen_tangram.tangram_element import Element


class Mask:
    """This class serves as STATE for searching.
    Member Variables:
        self.raw_mask: input image after preprocessing.
        self.grid: transform input image into grids, in which each pixel represents one 1x1 square.
        self.unit_length: ratio between self.raw_mask and self.grid.
        self.state: same shape as self.grid, in which where need to be placed has value == 0,
            where can't be placed has value == 255,
            other places has value == int(element_id) * 18
    """

    def __init__(self, raw_mask, tangram_s):
        # read in image and preprocess
        raw_mask = image_utils.binarize_image(raw_mask)
        self.unit_length, self.raw_mask, self.grid = \
            image_utils.segment_image(raw_mask, tangram_s)
        self.need_square = True if tangram_s % 2 == 0 else False

        # mark some elements of Tangram
        self.unused_element_ids = ['0', '1', '2', '3', '4', '5', '6',
                                   '7', '8', '9', '10', '11', '12']
        if not self.need_square:
            self.unused_element_ids.remove('0')
        self.used_elements = []
        self.state = copy.deepcopy(self.grid)

    def visualize_grid(self):
        """Show self.grid as gray scale image."""
        image_utils.show_gray_image(self.grid)

    def visualize_state(self):
        """Show self.grid as gray scale image."""
        image_utils.show_gray_image(self.state)

    def update(self, element):
        """Update self using element.
        Input:
            element: a dict which is
            {
                'element_id': element_id,
                'element': element,
                'x': x,
                'y': y,
                'angle': angle_idx
            }
        """
        new_element_id = element['element_id']
        assert new_element_id in self.unused_element_ids
        self.unused_element_ids.remove(new_element_id)
        self.used_elements.append(element)
        self.update_grid()
        # self.update_state()

    def update_grid(self):
        """Set where has been placed with value == 255."""
        element = self.used_elements[-1]
        for i in range(element['element'].area):
            self.grid[element['element'].coordinates[i][0],
                      element['element'].coordinates[i][1]] = 255

    def update_state(self):
        """Set places that have been filled by elements with value == element_id * 18."""
        for element in self.used_elements:
            element_id = int(element['element'].element_id)
            for i in range(element['element'].area):
                self.state[element['element'].coordinates[i][0],
                           element['element'].coordinates[i][1]] = 18 * (element_id + 1)

    def mask_is_same_mask(self, another_mask):
        """Judge whether this state has been searched."""
        return (self.grid == another_mask.grid).all()

    def element_is_inside_grid(self, element):
        """Judge whether every coordinate of element has a value == 0."""
        for i in range(element.area):
            x, y = element.coordinates[i]
            if x < 0 or x >= self.grid.shape[0] or y < 0 or y >= self.grid.shape[1]:
                return False
            if self.grid[x, y] != 0:
                return False
        return True

    def connectivity_area_is_valid(self, element):
        """The connectivity areas should have areas == 4 or 5."""
        grid = copy.deepcopy(self.grid)
        for i in range(element.area):
            grid[element.coordinates[i][0], element.coordinates[i][1]] = 255
        # reverse grid
        mask = np.zeros_like(grid, dtype=np.uint8)
        mask[grid == 0] = 255
        connectivity = skimage.measure.label(mask, connectivity=1)
        # get ares for all connectivity areas except for background
        bg_label = connectivity[0, 0]
        all_label = np.unique(connectivity.reshape(-1, ))
        areas = [(connectivity == label).sum()
                 for label in all_label if label != bg_label]
        # judge 4 or 5
        # square is the first element to be used so the remaining area should a multiple of 5!
        for area in areas:
            if area % 5 != 0:
                return False
        return True

    def element_is_valid(self, element):
        """Judge whether an element is valid."""
        # first judge index out of range
        if not self.element_is_inside_grid(element):
            return False

        # then judge areas of connected areas, should be 4 or 5
        if not self.connectivity_area_is_valid(element):
            return False

        return True

    def try_element(self, element_id, start_x, start_y, start_angle):
        """Try placing an element on the grid."""
        assert element_id in self.unused_element_ids
        assert start_x in range(self.grid.shape[0])
        assert start_y in range(self.grid.shape[1])

        # element prototype
        element = Element(element_id)

        for x in range(start_x, self.grid.shape[0]):
            for y in range(start_y, self.grid.shape[1]):
                for angle_idx in range(start_angle, ANGEL_NUM):
                    if self.grid[x, y] != 0:
                        break
                    element.set_points(p=np.array([x, y]), position=angle_idx)

                    if self.element_is_valid(element):
                        return {
                            'element_id': element_id,
                            'element': element,
                            'x': x,
                            'y': y,
                            'angle': angle_idx
                        }
                start_angle = 0
            start_y = 0
        return None


class TangramSolver:
    """This class contains all solvers for Tangram problem.
    Currently support only DFS
    """

    def __init__(self, raw_mask, tangram_s):
        self.mask = Mask(raw_mask, tangram_s)
        self.tangram_s = tangram_s
        self.need_square = self.mask.need_square
        self.need_element_num = int(math.ceil(tangram_s / 5.))
        self.past_state = []

    def DFS(self):
        if self.tangram_s % 5 not in [0, 4]:
            return None

        # possible combinations of elements
        candidates = itertools.combinations(self.mask.unused_element_ids,
                                            r=self.need_element_num)
        candidate_idx = 0
        element_ids = []
        state_list = []
        # iterate over all candidates
        while True:
            last_ids = copy.deepcopy(element_ids)
            try:
                element_ids = next(candidates)
            except StopIteration:
                return None

            # reuse some explored states to reduce searching time!
            # for example, if last_ids == [1, 2, 3, 4] and this _ids == [1, 2, 3, 5],
            # then all the states that use element [1, 2, 3] can be reused
            if len(last_ids) == 0:
                state_list = [self.mask]
                self.past_state = []
            else:
                same_num = 0
                for i in range(self.need_element_num):
                    if last_ids[i] == element_ids[i]:
                        same_num += 1
                    else:
                        break
                if same_num == 0:
                    state_list = [self.mask]
                    self.past_state = []
                else:
                    last_past_state = copy.deepcopy(self.past_state)
                    state_list = []
                    self.past_state = []
                    for state in last_past_state:
                        if len(state.used_elements) <= same_num:
                            self.past_state.append(copy.deepcopy(state))
                            if len(state.used_elements) == same_num:
                                state_list.append(copy.deepcopy(state))

                    del last_past_state, last_ids

            start_x = 0
            start_y = 0
            start_angle = 0

            candidate_idx += 1

            while len(state_list) > 0:
                cur_state = state_list.pop()
                next_element_id = element_ids[len(cur_state.used_elements)]

                result = cur_state.try_element(next_element_id, start_x, start_y, start_angle)

                while result is not None:
                    new_state = copy.deepcopy(cur_state)
                    new_state.update(result)

                    # judge whether this state has been searched before
                    same_flag = False
                    for past_state in self.past_state:
                        if new_state.mask_is_same_mask(past_state):
                            same_flag = True
                            break
                    if same_flag:
                        result = cur_state.try_element(next_element_id,
                                                       result['x'], result['y'], result['angle'] + 1)
                        continue
                    self.past_state.append(copy.deepcopy(new_state))

                    if len(new_state.used_elements) == self.need_element_num:
                        # success
                        return new_state

                    state_list.append(copy.deepcopy(new_state))
                    result = cur_state.try_element(next_element_id,
                                                   result['x'], result['y'], result['angle'] + 1)
                start_x = 0
                start_y = 0
                start_angle = 0
