"""
This file contains searching algorithm for the task.
Define STATE and TRANSFER function.
Currently I regard a mask with some pixels filled by Tangram elements as STATE,
    and Transfer function is adding another element on the mask.

Also, this file contains some hyperparameters for searching like
    the order of elements for searching, total number of states, e.g.
"""

import cv2
import copy
import math
import numpy as np

import any_tangram.image_utils as image_utils
from any_tangram.tangram_element import Point, Segment, Triangle, Square, Parallelogram
from any_tangram.config import ANGLE_NUM, ELEMENT_DICT, UNIT_LENGTH, SLIDING_INTERVAL, EXPECTED_RATIO, \
    PARALLELOGRAM_STATE, SQUARE_STATE, TRIANGLE_STATE, MAX_NUM


class Mask:
    """This class serves as the STATE of the searching algorithm.
    Member Variables:
        self.raw_mask: a binary mask representing the Tangram problem, typically a 2d np.array.
            Areas with value == 0 (black spaces) are places needs to be filled.
            Areas with value == 255 (white places) are places can't be filled.
            Will update this with the development of the problem solving,
                which is to say, if I place an element on somewhere, then I'll draw them black.
        self.tangram_s: the total area of a set of Tangram, typically 8.
        self.unit_length: the ratio between raw_mask and standard Tangram.
        self.unused_element_ids: ids of elements that haven't been used to fill the mask.
        self.used_elements: elements that have been used to fill the mask.
            Note that they are dicts with the following form:
            {
                'element_id': this_element_id,
                'element': element,
                'corner': corner_idx,
                'state': state_idx,
                'angle': angle_idx
            }
        self.state: a 2d np.array having the same shape as raw_mask.
            0 is where we need to fill, 255 is where we can't fill,
            128 is where has been filled with Tangram elements.
            Can use self.visualize_state() to see the problem solving procedure!
        self.error_threshold: used to handle non-integer length of elements and other minor errors.
    The main difference between any_tangram problem and normal_tangram problem is that,
        here we no longer consider corners and segments of the raw_mask since it's okay to slightly
        extend elements out of the mask.
    We relax the constrain of element_out_of_mask, but more strict about element_intersect_element.
    Also, since we no longer search along vertex and corners but rather every pixel,
        we no longer need different states for elements!
    """

    def __init__(self, raw_mask, tangram_s=8.):
        # read in image and get unit length
        self.raw_mask = copy.deepcopy(raw_mask)
        self.raw_mask = image_utils.binarize_image(self.raw_mask)
        self.tangram_s = tangram_s
        unit_length = image_utils.get_unit_length(self.raw_mask, self.tangram_s) * math.sqrt(EXPECTED_RATIO)
        # for smaller searching space
        self.unit_length = min(UNIT_LENGTH, unit_length)
        self.expected_ratio = EXPECTED_RATIO
        self.raw_mask = image_utils.resize_image_to_unit_length(self.raw_mask, self.unit_length,
                                                                self.tangram_s, self.expected_ratio)
        self.element_out_of_mask_error_threshold = self.unit_length * 0.15
        self.element_intersect_element_error_threshold = self.unit_length * 0.1
        self.element_coincide_element_error_threshold = SLIDING_INTERVAL / 2.

        # mark some elements of Tangram
        self.unused_element_ids = ['1', '2', '3', '4', '5', '6', '7']
        self.used_elements = []
        self.state = copy.deepcopy(raw_mask)

    def update(self, element):
        """Update self by input element.
            1.remove unused_element_ids[0]
            2.append used_elements
            3.update state
            4.update raw_mask
        Input:
            element: {
                'element_id': this_element_id,
                'element': element,
                'x': x,
                'y': y,
                'state': state_idx,
                'angle': angle_idx
            }
        """
        new_element_id = element['element_id']
        assert new_element_id in self.unused_element_ids
        self.unused_element_ids.remove(new_element_id)
        self.used_elements.append(element)
        self.update_mask()
        self.update_state()

    def update_mask(self):
        """Fill used elements on self.raw_mask with value 255 (make them white).
        Which is to say, not allowed for placing elements here.
        """
        self.raw_mask = \
            image_utils.draw_polygon(self.raw_mask, self.used_elements[-1]['element'], color=255)

    def update_state(self):
        """Fill self.state with 0, 128 and 255."""
        self.state = copy.deepcopy(self.raw_mask)
        for element in self.used_elements:
            self.state = \
                image_utils.draw_polygon(self.state, element['element'], color=128)

    def visualize_mask(self):
        """Show self.raw_mask as gray scale image."""
        image_utils.show_gray_image(self.raw_mask)

    def visualize_state(self):
        """Show self.state as gray scale image."""
        image_utils.show_gray_image(self.state)

    def mask_is_equal_mask(self, another_mask):
        """Judge whether this mask is the same one.
        For example, although I place a triangle according to different corners,
            it's possible that they looks the same via some rotations or so.
        """
        # number of used elements is not the same
        if len(self.unused_element_ids) != len(another_mask.unused_element_ids):
            return False

        # judge whether each used_element has a coinciding used_element
        flag = False
        for element1 in self.used_elements:
            for element2 in another_mask.used_elements:
                e1 = element1['element']
                e2 = element2['element']
                if e1.element_is_coincide_element(e2, self.element_coincide_element_error_threshold):
                    flag = True
                    break
            if not flag:
                return False
            flag = False
        return True

    def point_is_inside_mask(self, another_point):
        """Judge whether a point is within self.raw_mask."""
        # only areas that have a value == 255 can't be placed
        return image_utils. \
            point_is_inside_mask(self.raw_mask, another_point, self.element_out_of_mask_error_threshold, value=0)

    def element_keypoint_is_inside_mask(self, another_element):
        """Judge whether all the keypoints of an element are within place_able areas of self.raw_mask.
        Currently I'll examine vertices, the midpoint of the element, the midpoint of every segments.
        """
        # vertex
        for point in another_element.points:
            if not self.point_is_inside_mask(point):
                return False
        # middle point
        if not self.point_is_inside_mask(another_element.midpoint):
            return False
        # midpoint of segments
        for i in range(another_element.point_num):
            if not self.point_is_inside_mask(another_element.segments[i].midpoint):
                return False
        return True

    def element_is_valid(self, element):
        """Judge whether a place of an element is valid on the mask.
        This function is built to embrace all kinds of elements.
        """
        # now I have an element with coordinates vertices
        # judge whether it's valid by:
        # 1.all element.points should be within self.raw_mask
        # 2.all element.segments should not intersect with self.used_elements.segments
        if not self.element_keypoint_is_inside_mask(element):
            return False

        flag = False
        for used_element in self.used_elements:
            if element.element_is_intersect_element(used_element['element'],
                                                    self.element_intersect_element_error_threshold):
                flag = True
                break

        if flag:
            return False
        return True

    def try_triangle(self, size, start_x, start_y, start_state, start_angle):
        """Try to place a triangle on the mask.
        Input:
            size: 0 means small, 1 means medium, 2 means large
        """
        assert size in range(3)
        assert start_state in range(1)

        if size == 0:
            assert '7' in self.unused_element_ids
            if '6' in self.unused_element_ids:
                this_element_id = '6'
            else:
                this_element_id = '7'
        elif size == 1:
            assert '5' in self.unused_element_ids
            this_element_id = '5'
        else:
            assert '2' in self.unused_element_ids
            if '1' in self.unused_element_ids:
                this_element_id = '1'
            else:
                this_element_id = '2'

        # triangle prototype
        triangle = Triangle(this_element_id, size, self.unit_length)

        for x in range(start_x, self.raw_mask.shape[0], SLIDING_INTERVAL):
            for y in range(start_y, self.raw_mask.shape[1], SLIDING_INTERVAL):
                p = Point(x=x, y=y)
                if not self.point_is_inside_mask(p):
                    continue
                for state_idx in range(start_state, 1):  # no longer need different element state
                    for angle_idx in range(start_angle, ANGLE_NUM):
                        triangle.set_points(p=p,
                                            position=state_idx * ANGLE_NUM + angle_idx)

                        # judge whether this position is valid
                        if self.element_is_valid(triangle):
                            return {
                                'element_id': this_element_id,
                                'element': triangle,
                                'x': x,
                                'y': y,
                                'state': state_idx,
                                'angle': angle_idx
                            }

                    start_angle = 0
                start_state = 0
            start_y = 0
        return None

    def try_square(self, size, start_x, start_y, start_state, start_angle):
        """Try to place a square on the mask."""
        assert start_state in range(1)
        assert '3' in self.unused_element_ids

        # square prototype
        square = Square(element_id='3', unit_length=self.unit_length)

        for x in range(start_x, self.raw_mask.shape[0], SLIDING_INTERVAL):
            for y in range(start_y, self.raw_mask.shape[1], SLIDING_INTERVAL):
                p = Point(x=x, y=y)
                if not self.point_is_inside_mask(p):
                    continue
                # there is only one state for square!
                for state_idx in range(start_state, 1):  # no longer need different element state
                    for angle_idx in range(start_angle, ANGLE_NUM):
                        square.set_points(p=p,
                                          position=state_idx * ANGLE_NUM + angle_idx)

                        # judge whether this position is valid
                        if self.element_is_valid(square):
                            return {
                                'element_id': '3',
                                'element': square,
                                'x': x,
                                'y': y,
                                'state': state_idx,
                                'angle': angle_idx
                            }

                    start_angle = 0
                start_state = 0
            start_y = 0
        return None

    def try_parallelogram(self, size, start_x, start_y, start_state, start_angle):
        """Try to place a parallelogram on the mask."""
        assert start_state in range(1)
        assert '4' in self.unused_element_ids

        # parallelogram prototype
        parallelogram = Parallelogram(element_id='4', unit_length=self.unit_length)

        for x in range(start_x, self.raw_mask.shape[0]):
            for y in range(start_y, self.raw_mask.shape[1]):
                p = Point(x=x, y=y)
                if not self.point_is_inside_mask(p):
                    continue
                for state_idx in range(start_state, 1):  # no longer need different element state
                    for angle_idx in range(start_angle, ANGLE_NUM):
                        parallelogram.set_points(p=p,
                                                 position=state_idx * ANGLE_NUM + angle_idx)

                        # judge whether this position is valid
                        if self.element_is_valid(parallelogram):
                            return {
                                'element_id': '4',
                                'element': parallelogram,
                                'x': x,
                                'y': y,
                                'state': state_idx,
                                'angle': angle_idx
                            }

                    start_angle = 0
                start_state = 0
            start_y = 0
        return None


class TangramSolver:
    """This class contains all solvers for Tangram problem.
    Currently support only DFS
    """

    def __init__(self, raw_mask, tangram_s=8.):
        self.mask = Mask(raw_mask, tangram_s)
        self.raw_mask = copy.deepcopy(self.mask.raw_mask)
        self.past_state = []
        # store all possible solutions and select the max IOU one to return
        self.valid_solution = []

    def DFS(self):
        state_list = [self.mask]

        start_x = 0
        start_y = 0
        start_state = 0
        start_angle = 0
        size = 0
        while len(state_list) > 0:
            cur_state = state_list.pop()
            element_id = cur_state.unused_element_ids[0]
            if element_id == '1' or element_id == '2':
                size = 2
                try_function = cur_state.try_triangle
            elif element_id == '5':
                size = 1
                try_function = cur_state.try_triangle
            elif element_id == '6' or element_id == '7':
                size = 0
                try_function = cur_state.try_triangle
            elif element_id == '3':
                try_function = cur_state.try_square
            else:
                try_function = cur_state.try_parallelogram

            result = try_function(size, start_x, start_y, start_state, start_angle)

            while result is not None:
                # push this new state into state_list
                new_state = copy.deepcopy(cur_state)
                new_state.update(result)

                # judge whether this state has been pushed into state_list
                same_flag = False
                for past_state in self.past_state:
                    if new_state.mask_is_equal_mask(past_state):
                        # this state has been pushed into state_list
                        # shouldn't re-push!
                        same_flag = True
                        break

                if same_flag:
                    result = try_function(size, result['x'], result['y'], result['state'], result['angle'] + 1)
                    continue

                self.past_state.append(copy.deepcopy(new_state))

                if len(new_state.unused_element_ids) == 0:
                    # store this valid solution
                    self.valid_solution.append(copy.deepcopy(new_state))

                    if len(self.valid_solution) >= MAX_NUM:
                        # break out of while loop and go to IOU comparing
                        state_list = []
                        break

                    result = try_function(size, result['x'], result['y'], result['state'], result['angle'] + 1)
                    continue

                state_list.append(copy.deepcopy(new_state))
                result = try_function(size, result['x'], result['y'], result['state'], result['angle'] + 1)

            # has pushed some new states into state_list from current state
            start_x = 0
            start_y = 0
            start_state = 0
            start_angle = 0
            size = 0

        if len(self.valid_solution) == 0:
            return None
        else:
            solutions = []
            for solution in self.valid_solution:
                solutions.append(
                    image_utils.draw_polygons_on_mask(np.ones_like(self.raw_mask) * 255,
                                                      solution.used_elements, color=0)
                )
            ious = image_utils.compute_iou(self.raw_mask, np.stack(solutions))
            max_iou = ious.max()
            max_idx = np.argmax(ious)

            return self.valid_solution[max_idx]
