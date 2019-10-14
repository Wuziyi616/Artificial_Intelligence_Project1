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
import numpy as np

import normal_tangram.image_utils as image_utils
from normal_tangram.tangram_element import Point, Segment, Triangle, Square, Parallelogram
from normal_tangram.config import ANGLE_NUM, ELEMENT_DICT, \
    PARALLELOGRAM_STATE, SQUARE_STATE, TRIANGLE_STATE


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
        self.connected_area_num: number of connected areas in current self.raw_mask.
        self.corner_num: number of corners.
        self.corners, self.segments: used to validate new elements.
    """

    def __init__(self, raw_mask, tangram_s=8.):
        # read in image and get unit length
        self.raw_mask = copy.deepcopy(raw_mask)
        self.raw_mask = image_utils.binarize_image(self.raw_mask)
        self.tangram_s = tangram_s
        self.unit_length = image_utils.get_unit_length(self.raw_mask, standard_s=tangram_s)
        self.error_threshold = self.unit_length * 0.15
        # preprocess input image
        self.eliminate_gaps()

        # init corners and segments
        self.connected_area_num = 0
        self.corner_num = 0
        self.corners = []
        self.segments = []
        self.max_x = 0
        self.max_y = 0
        self.update_corners_and_segments()

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
            5.update corners and segments
        Input:
            element: {
                'element_id': this_element_id,
                'element': element,
                'corner': corner_idx,
                'state': state_idx,
                'angle': angle_idx
            }
        """
        new_element_id = element['element_id']
        assert new_element_id in self.unused_element_ids
        self.unused_element_ids.remove(new_element_id)
        self.used_elements.append(element)
        self.update_mask()
        self.update_corners_and_segments()

    def update_mask(self):
        """Fill used elements on self.raw_mask with value 255 (make them white).
        Which is to say, not allowed for placing elements here.
        """
        self.raw_mask = \
            image_utils.draw_polygon(self.raw_mask, self.used_elements[-1]['element'], color=255)
        self.eliminate_gaps()

    def eliminate_gaps(self):
        """erode and dilate to eliminate small gaps or holes."""
        # first eliminate small black areas
        self.raw_mask = cv2.dilate(self.raw_mask, None, int(self.error_threshold * 0.1))
        self.raw_mask = cv2.erode(self.raw_mask, None, int(self.error_threshold * 0.1))
        # then eliminate small white areas
        self.raw_mask = cv2.erode(self.raw_mask, None, int(self.error_threshold * 0.1))
        self.raw_mask = cv2.dilate(self.raw_mask, None, int(self.error_threshold * 0.1))

    def visualize_mask(self):
        """Show self.raw_mask as gray scale image."""
        image_utils.show_gray_image(self.raw_mask)

    def update_corners_and_segments(self):
        """Update self.corners and self.segments using self.raw_mask."""
        # first set all these member variables to 0
        self.connected_area_num = 0
        self.corner_num = 0
        self.corners = []
        self.segments = []
        max_coordinate = [0., 0.]

        # all_corners: [[[x11, y11], [x12, y12], ...], [[x21, y21], [x22, y22], ...], ...]
        all_corners = image_utils.detect_corners(self.raw_mask, 0.5 * self.error_threshold)
        self.connected_area_num = len(all_corners)

        for i in range(self.connected_area_num):
            temp_coor = np.max(np.stack(all_corners[i], axis=0), axis=0)
            max_coordinate[0] = max(max_coordinate[0], temp_coor[0])
            max_coordinate[1] = max(max_coordinate[1], temp_coor[1])

            num = len(all_corners[i])

            # store corners
            self.corners += [
                Point(x=all_corners[i][j][0], y=all_corners[i][j][1]) for j in range(num)
            ]

            # store segments, each connected area is separated
            self.segments += [
                Segment(p1=self.corners[j + self.corner_num],
                        p2=self.corners[(j + 1) % num + self.corner_num])
                for j in range(num)
            ]
            self.corner_num += num

        # also store the outer lines of the total input image
        self.segments += [
            Segment(p1=Point(x=0., y=0.),
                    p2=Point(x=0., y=self.raw_mask.shape[1] - 1.)),
            Segment(p1=Point(x=0., y=0.),
                    p2=Point(x=self.raw_mask.shape[0] - 1., y=0.)),
            Segment(p1=Point(x=self.raw_mask.shape[0] - 1., y=self.raw_mask.shape[1] - 1.),
                    p2=Point(x=0., y=self.raw_mask.shape[1] - 1.)),
            Segment(p1=Point(x=self.raw_mask.shape[0] - 1., y=self.raw_mask.shape[1] - 1.),
                    p2=Point(x=self.raw_mask.shape[0] - 1., y=0.))
        ]

        # update max_x and max_y of areas that needs to be filled
        self.max_x, self.max_y = max_coordinate

    def update_state(self):
        """Fill self.state with 0, 128 and 255."""
        self.state = copy.deepcopy(self.raw_mask)
        for element in self.used_elements:
            self.state = \
                image_utils.draw_polygon(self.state, element['element'], color=128)

    def visualize_state(self):
        """Show self.state as gray scale image."""
        image_utils.show_gray_image(self.state)

    def visualize_corners(self):
        """Show self.corners on raw_mask."""
        mask = copy.deepcopy(self.raw_mask)
        mask = np.stack([mask, mask, mask], axis=-1)
        for corner in self.corners:
            x, y = corner.get_coordinate()
            x, y = int(x), int(y)
            mask[x - 5:x + 5, y - 5:y + 5] = (255, 0, 0)
        image_utils.show_image(mask)

    def visualize_segments(self):
        """Show self.segments on raw_mask."""
        mask = copy.deepcopy(self.raw_mask)
        mask = np.stack([mask, mask, mask], axis=-1)
        for segment in self.segments:
            x1, y1 = segment.p1.get_coordinate()
            x2, y2 = segment.p2.get_coordinate()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(mask, (y1, x1), (y2, x2),
                     color=(0, 0, 255), thickness=3)
        image_utils.show_image(mask)

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
                if e1.element_is_coincide_element(e2, self.error_threshold):
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
            point_is_inside_mask(self.raw_mask, another_point, self.error_threshold, value=0)

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
        # 2.all element.segments should not intersect with self.segments
        # 3.all element.segments should not intersect with self.used_elements.segments
        if not self.element_keypoint_is_inside_mask(element):
            return False

        # check whether new element intersect with self.raw_mask
        if element.element_is_intersect_element(self, self.error_threshold):
            return False

        flag = False
        for used_element in self.used_elements:
            if element.element_is_intersect_element(used_element['element'], self.error_threshold):
                flag = True
                break

        if flag:
            return False
        return True

    def try_triangle(self, size, start_corner, start_state, start_angle):
        """Try to place a triangle on the mask.
        Input:
            size: 0 means small, 1 means medium, 2 means large
        """
        assert size in range(3)
        assert start_corner in range(self.corner_num)
        assert start_state in range(TRIANGLE_STATE)

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

        for corner_idx in range(start_corner, self.corner_num):
            for state_idx in range(start_state, TRIANGLE_STATE):
                for angle_idx in range(start_angle, ANGLE_NUM):
                    triangle.set_points(p=self.corners[corner_idx],
                                        position=state_idx * ANGLE_NUM + angle_idx)

                    # judge whether this position is valid
                    if self.element_is_valid(triangle):
                        return {
                            'element_id': this_element_id,
                            'element': triangle,
                            'corner': corner_idx,
                            'state': state_idx,
                            'angle': angle_idx
                        }

                start_angle = 0
            start_state = 0
        return None

    def try_square(self, size, start_corner, start_state, start_angle):
        """Try to place a square on the mask."""
        assert start_corner in range(self.corner_num)
        assert start_state in range(SQUARE_STATE)
        assert '3' in self.unused_element_ids

        # square prototype
        square = Square(element_id='3', unit_length=self.unit_length)

        for corner_idx in range(start_corner, self.corner_num):
            # there is only one state for square!
            for state_idx in range(start_state, SQUARE_STATE):
                for angle_idx in range(start_angle, ANGLE_NUM):
                    square.set_points(p=self.corners[corner_idx],
                                      position=state_idx * ANGLE_NUM + angle_idx)

                    # judge whether this position is valid
                    if self.element_is_valid(square):
                        return {
                            'element_id': '3',
                            'element': square,
                            'corner': corner_idx,
                            'state': state_idx,
                            'angle': angle_idx
                        }

                start_angle = 0
            start_state = 0
        return None

    def try_parallelogram(self, size, start_corner, start_state, start_angle):
        """Try to place a parallelogram on the mask."""
        assert start_corner in range(self.corner_num)
        assert start_state in range(PARALLELOGRAM_STATE)
        assert '4' in self.unused_element_ids

        # parallelogram prototype
        parallelogram = Parallelogram(element_id='4', unit_length=self.unit_length)

        for corner_idx in range(start_corner, self.corner_num):
            for state_idx in range(start_state, PARALLELOGRAM_STATE):
                for angle_idx in range(start_angle, ANGLE_NUM):
                    parallelogram.set_points(p=self.corners[corner_idx],
                                             position=state_idx * ANGLE_NUM + angle_idx)

                    # judge whether this position is valid
                    if self.element_is_valid(parallelogram):
                        return {
                            'element_id': '4',
                            'element': parallelogram,
                            'corner': corner_idx,
                            'state': state_idx,
                            'angle': angle_idx
                        }

                start_angle = 0
            start_state = 0
        return None


class TangramSolver:
    """This class contains all solvers for Tangram problem.
    Currently support only DFS
    """

    def __init__(self, raw_mask, tangram_s=8.):
        self.mask = Mask(raw_mask, tangram_s)
        self.past_state = []

    def DFS(self):
        state_list = [self.mask]

        start_corner = 0
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

            result = try_function(size, start_corner, start_state, start_angle)

            while result is not None:
                # refine the coordinates of new element
                result['element'].constrain(cur_state.max_x, cur_state.max_y)

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
                    result = try_function(size, result['corner'], result['state'], result['angle'] + 1)
                    continue

                self.past_state.append(copy.deepcopy(new_state))

                if len(new_state.unused_element_ids) == 0:
                    # succeed
                    return new_state

                state_list.append(copy.deepcopy(new_state))
                result = try_function(size, result['corner'], result['state'], result['angle'] + 1)

            # has pushed some new states into state_list from current state
            start_corner = 0
            start_state = 0
            start_angle = 0
            size = 0

        return None
