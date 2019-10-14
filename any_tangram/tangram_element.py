"""
This file contains class definition for elements in a set of Tangram.
Typically there are Triangles, Squares and Parallelograms.
Maybe add more elements in the future for 13, 14, 15... Tangram.

Also, this file defines some hyperparameters for calculating.
"""

import copy
from math import sin, cos
import numpy as np

import any_tangram.utils as utils
from any_tangram.config import PI, SQRT_2, ANGLE_NUM, STANDARD_SIZE, \
    ELEMENT_DICT, TRIANGLE_STATE, SQUARE_STATE, PARALLELOGRAM_STATE


class Point:

    def __init__(self, x=0., y=0.):
        self.x = float(x)
        self.y = float(y)

    def __eq__(self, other):
        """Define the equal operator between 2 points.
        Return True when the coordinates of 2 points are the same.
        """
        if self.x == other.x and self.y == other.y:
            return True
        return False

    def __add__(self, other):
        """Define the add operator between 2 points."""
        return Point(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other):
        """Define the sub operator between 2 points.
        Can be used to calculate the vector.
        """
        return Point(x=self.x - other.x, y=self.y - other.y)

    def constrain(self, max_x, max_y):
        """Use max_x, max_y to keep point within mask."""
        self.x = max(min(self.x, max_x), 0)
        self.y = max(min(self.y, max_y), 0)

    def get_coordinate(self):
        return [self.x, self.y]

    def point_is_coincide_point(self, another_point, threshold):
        """If the distance between 2 points is within an error threshold,
        then we say they coincide with each other.
        """
        if utils.get_distance_point_to_point(self, another_point) < threshold:
            return True
        return False

    def display(self):
        """For debug."""
        print('coordinate:', self.get_coordinate())


class Segment:

    def __init__(self, p1, p2):
        self.p1 = copy.deepcopy(p1)
        self.p2 = copy.deepcopy(p2)
        self.midpoint = Point(x=(p1.x + p2.x) / 2., y=(p1.y + p2.y) / 2.)
        self.vec = self.p2 - self.p1

    def constrain(self, max_x, max_y):
        """Use max_x, max_y to keep point within mask."""
        self.p1.constrain(max_x, max_y)
        self.p2.constrain(max_x, max_y)
        self.midpoint = Point(x=(self.p1.x + self.p2.x) / 2., y=(self.p1.y + self.p2.y) / 2.)
        self.vec = self.p2 - self.p1

    def point_is_on_segment(self, another_point, threshold):
        """Judge whether a point is ON this segment.
        Endpoint of the segment is considered on here.
        threshold is a term that allows for tiny error (caused by non-integer of coordinates)
        """
        dist1 = utils.get_distance_point_to_point(self.p1, another_point)
        dist2 = utils.get_distance_point_to_point(self.p2, another_point)
        if dist1 < threshold or dist2 < threshold:
            # the point is one of the endpoints of this segment
            return True

        v1 = self.p1 - another_point
        v2 = self.p2 - another_point
        if abs(utils.inner_product(v1, v2) + dist1 * dist2) < threshold ** 2:
            return True
        return False

    def __segment_is_intersect_segment(self, another_segment, threshold):
        """Judge intersect using cross product.
        Won't be used outside the class so set it as private method.
        """
        # 2 segments are not parallel
        v1 = another_segment.p1 - self.p1
        v2 = another_segment.p2 - self.p1
        if utils.cross_product(v1, self.vec) * utils.cross_product(v2, self.vec) < -(threshold ** 4):
            return True
        return False

    def segment_is_cross_segment(self, another_segment, threshold):
        """Judge whether this segment is cross over another segment.
        Use mutual cross product to judge.
        Any parallel or even coinciding segments are not considered intersect here.
        __|__ is not considered intersect too. Only --|-- is considered intersect!
        threshold is a term that allows for tiny error (caused by non-integer of coordinates)
        """
        if self.point_is_on_segment(another_segment.p1, threshold) or \
                self.point_is_on_segment(another_segment.p2, threshold) or \
                another_segment.point_is_on_segment(self.p1, threshold) or \
                another_segment.point_is_on_segment(self.p2, threshold):
            # one point of one segment is on another segment
            return False

        if abs(self.vec.x * another_segment.vec.y - self.vec.y * another_segment.vec.x) < threshold ** 2:
            # 2 segments are parallel
            return False

        if self.__segment_is_intersect_segment(another_segment, threshold) and \
                another_segment.__segment_is_intersect_segment(self, threshold):
            return True
        return False

    def display(self):
        """For debug."""
        print('point1:', self.p1.get_coordinate())
        print('point2:', self.p2.get_coordinate())


class Element:
    """Abstract class for all elements."""

    def __init__(self, element_id, unit_length):
        """Actually we should init the class using init function."""
        self.points = None
        self.point_num = 0
        self.midpoint = None
        self.segments = []
        self.element_id = element_id
        self.unit_length = unit_length
        self.position = None
        self.state = None
        self.area = STANDARD_SIZE[ELEMENT_DICT[self.element_id]][0] * (self.unit_length ** 2)
        self.length = STANDARD_SIZE[ELEMENT_DICT[self.element_id]][-1] * self.unit_length
        self.size = 0

    def init(self, points):
        """Init element by setting points etc."""
        self.points = copy.deepcopy(points)
        self.point_num = len(points)
        self.midpoint = self.get_midpoint()
        self.segments = [
            Segment(self.points[i], self.points[(i + 1) % self.point_num]) for i in range(self.point_num)
        ]
        # add segments between every midpoints
        # in order to judge intersection more accurately
        self.segments += [
            Segment(p1=self.segments[i].midpoint,
                    p2=self.segments[(i + 1) % self.point_num].midpoint)
            for i in range(self.point_num)
        ]

    def set_points(self, p, position):
        """Calculate the coordinates for vertices of this element.
        Input:
            p: a point with coordinate
            position: can be used to determine angle
        """
        pass

    def get_midpoint(self):
        """Get the midpoint of the whole element by averaging all x and y coordinates."""
        x = 0.
        y = 0.
        for point in self.points:
            x += (point.x / self.point_num)
            y += (point.y / self.point_num)

        return Point(x=x, y=y)

    def constrain(self, max_x, max_y):
        """Use max_x, max_y to keep point within mask."""
        for point in self.points:
            point.constrain(max_x, max_y)
        for segment in self.segments:
            segment.constrain(max_x, max_y)

    def element_is_coincide_element(self, another_element, threshold):
        """Judge whether two elements coincide."""
        if type(self) != type(another_element):
            return False

        if type(self) == Triangle and type(another_element) == Triangle:
            if self.size != another_element.size:
                return False

        num = self.point_num
        flag = np.array([False for _ in range(num)])
        for i in range(num):
            for j in range(num):
                if self.points[i].point_is_coincide_point(another_element.points[j], threshold):
                    flag[i] = True
                    break
        if flag.all():
            return True
        return False

    def element_is_intersect_element(self, another_element, threshold):
        """Judge whether an element intersects with this element."""
        if self.element_is_coincide_element(another_element, threshold):
            return True
        for segment1 in self.segments:
            for segment2 in another_element.segments:
                if segment1.segment_is_cross_segment(segment2, threshold):
                    return True
        return False

    def point_is_inside_element(self, another_point, threshold):
        """Judge whether a point is within this element.
        A point on the edges of an element is not considered INSIDE here.
        """
        for segment in self.segments[:self.point_num]:
            if segment.point_is_on_segment(another_point, threshold):
                return False
        area = 0.
        for i in range(self.point_num):
            area += \
                utils.triangle_area([self.points[i], self.points[(i + 1) % self.point_num], another_point])
        if abs(area - self.area) < threshold ** 2:
            return True
        return False

    def display(self):
        """For debug."""
        print('element id:', self.element_id)
        print('state:', self.state)
        print('position:', self.position)
        print('length:', self.length)
        for i in range(self.point_num):
            print('point', i, 'coordinate:', self.points[i].get_coordinate())


class Triangle(Element):
    """Class for triangle. Note that point0 is right angle vertex."""

    def __init__(self, element_id, size=0, unit_length=0.):
        """
        Input:
            size: 0 means small, 1 means medium, 2 means large
        """
        super(Triangle, self).__init__(element_id, unit_length)

        assert size in range(3)
        self.size = size

    def init(self, points):
        """For triangle, I think we should add more segments."""
        super(Triangle, self).init(points)
        # add more segments for large triangle
        if self.element_id in ['1', '2']:
            self.segments += [
                Segment(p1=self.points[i], p2=self.segments[(i + 1) % 3].midpoint)
                for i in range(3)
            ]

    def set_points(self, p, position):
        """Calculate the coordinates for vertices of this triangle.
        Input:
            p: a point with its coordinate.
                Maybe p0 or p1 or p2, judging by position // ANGLE_NUM
            position: can be used to determine angle and type of p
        """
        # totally 3 states for triangles
        self.state = position // ANGLE_NUM
        assert self.state in range(TRIANGLE_STATE)
        position %= ANGLE_NUM
        self.position = position
        angle = 2. * PI * position / ANGLE_NUM

        if self.state == 0:
            self.__set_points_via_p0(p, angle)
        elif self.state == 1:
            self.__set_points_via_p1(p, angle)
        else:
            self.__set_points_via_p2(p, angle)

    def __set_points_via_p0(self, p0, angle):
        """Calculating the coordinates of this triangle's vertices using p0."""
        x0, y0 = p0.get_coordinate()
        p1 = Point(x=x0 + self.length * sin(angle),
                   y=y0 + self.length * cos(angle))
        p2 = Point(x=x0 + self.length * cos(angle),
                   y=y0 - self.length * sin(angle))
        self.init([p0, p1, p2])

    def __set_points_via_p1(self, p1, angle):
        """Calculating the coordinates of this triangle's vertices using p1."""
        x1, y1 = p1.get_coordinate()
        p0 = Point(x=x1 + self.length * sin(0.25 * PI + angle),
                   y=y1 + self.length * cos(0.25 * PI + angle))
        p2 = Point(x=x1 + SQRT_2 * self.length * sin(angle),
                   y=y1 + SQRT_2 * self.length * cos(angle))
        self.init([p0, p1, p2])

    def __set_points_via_p2(self, p2, angle):
        """Calculating the coordinates of this triangle's vertices using p2."""
        x2, y2 = p2.get_coordinate()
        p0 = Point(x=x2 + self.length * sin(angle),
                   y=y2 + self.length * cos(angle))
        p1 = Point(x=x2 + SQRT_2 * self.length * cos(0.25 * PI - angle),
                   y=y2 + SQRT_2 * self.length * sin(0.25 * PI - angle))
        self.init([p0, p1, p2])


class Square(Element):
    """Class for triangle."""

    def __init__(self, element_id, unit_length=0.):
        super(Square, self).__init__(element_id, unit_length)

    def set_points(self, p, position):
        """Calculate the coordinates for vertices of this square.
        Input:
            p: a point with coordinate
            position: can be used to determine angle
        """
        # totally 1 state for square
        self.state = position // ANGLE_NUM
        assert self.state in range(SQUARE_STATE)
        position %= ANGLE_NUM
        self.position = position
        angle = 2. * PI * position / ANGLE_NUM

        self.__set_points_via_p0(p, angle)

    def __set_points_via_p0(self, p0, angle):
        """Calculating the coordinates of this triangle's vertices using p0."""
        x0, y0 = p0.get_coordinate()
        p1 = Point(x=x0 + self.length * sin(angle),
                   y=y0 + self.length * cos(angle))
        p2 = Point(x=x0 + self.length * sin(angle) + self.length * cos(angle),
                   y=y0 + self.length * cos(angle) - self.length * sin(angle))
        p3 = Point(x=x0 + self.length * cos(angle),
                   y=y0 - self.length * sin(angle))
        self.init([p0, p1, p2, p3])


class Parallelogram(Element):
    """Class for Parallelogram. Note that point0 is the acute angle vertex."""

    def __init__(self, element_id, unit_length=0.):
        super(Parallelogram, self).__init__(element_id, unit_length)

    def set_points(self, p, position):
        """Calculate the coordinates for vertices of this parallelogram.
        Input:
            p: a point with its coordinate.
                Maybe p0 or p1, judging by position // ANGLE_NUM % 2
            position: can be used to determine angle and type of p
                Also, position // ANGLE_NUM // 2 is INVERSE STATE
        """
        # totally 4 states for parallelogram
        self.state = position // ANGLE_NUM
        assert self.state in range(PARALLELOGRAM_STATE)
        position %= ANGLE_NUM
        self.position = position
        angle = 2. * PI * position / ANGLE_NUM

        if self.state == 0:
            self.__set_points_via_p0(p, angle)
        elif self.state == 1:
            self.__set_points_via_p1(p, angle)
        elif self.state == 2:
            self.__set_points_via_inverse_p0(p, angle)
        else:
            self.__set_points_via_inverse_p1(p, angle)

    def __set_points_via_p0(self, p0, angle):
        """Calculating the coordinates of this parallelogram's vertices using p0."""
        x0, y0 = p0.get_coordinate()
        p1 = Point(x=x0 + SQRT_2 * self.length * sin(angle),
                   y=y0 + SQRT_2 * self.length * cos(angle))
        p2 = Point(x=x0 + SQRT_2 * self.length * sin(angle) + self.length * cos(0.25 * PI - angle),
                   y=y0 + SQRT_2 * self.length * cos(angle) + self.length * sin(0.25 * PI - angle))
        p3 = Point(x=x0 + self.length * cos(0.25 * PI - angle),
                   y=y0 + self.length * sin(0.25 * PI - angle))
        self.init([p0, p1, p2, p3])

    def __set_points_via_p1(self, p1, angle):
        """Calculating the coordinates of this parallelogram's vertices using p1."""
        x1, y1 = p1.get_coordinate()
        p0 = Point(x=x1 + SQRT_2 * self.length * sin(0.25 * PI - angle),
                   y=y1 - SQRT_2 * self.length * cos(0.25 * PI - angle))
        p2 = Point(x=x1 + self.length * sin(angle),
                   y=y1 + self.length * cos(angle))
        p3 = Point(x=x1 + SQRT_2 * self.length * sin(0.25 * PI - angle) + self.length * sin(angle),
                   y=y1 - SQRT_2 * self.length * cos(0.25 * PI - angle) + self.length * cos(angle))
        self.init([p0, p1, p2, p3])

    def __set_points_via_inverse_p0(self, p0, angle):
        """Calculating the coordinates of this parallelogram's vertices using inverse_p0."""
        x0, y0 = p0.get_coordinate()
        p1 = Point(x=x0 + self.length * sin(angle),
                   y=y0 + self.length * cos(angle))
        p2 = Point(x=x0 + self.length * sin(angle) + SQRT_2 * self.length * cos(0.25 * PI - angle),
                   y=y0 + self.length * cos(angle) + SQRT_2 * self.length * sin(0.25 * PI - angle))
        p3 = Point(x=x0 + SQRT_2 * self.length * cos(0.25 * PI - angle),
                   y=y0 + SQRT_2 * self.length * sin(0.25 * PI - angle))
        self.init([p0, p1, p2, p3])

    def __set_points_via_inverse_p1(self, p1, angle):
        """Calculating the coordinates of this parallelogram's vertices using inverse_p1."""
        x1, y1 = p1.get_coordinate()
        p0 = Point(x=x1 + self.length * sin(0.25 * PI - angle),
                   y=y1 - self.length * cos(0.25 * PI - angle))
        p2 = Point(x=x1 + SQRT_2 * self.length * sin(angle),
                   y=y1 + SQRT_2 * self.length * cos(angle))
        p3 = Point(x=x1 + self.length * sin(0.25 * PI - angle) + SQRT_2 * self.length * sin(angle),
                   y=y1 - self.length * cos(0.25 * PI - angle) + SQRT_2 * self.length * cos(angle))
        self.init([p0, p1, p2, p3])
