"""This File contains some utility functions"""

import math


def get_distance_point_to_point(p1, p2):
    """Get the distance between 2 points."""

    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def get_distance_point_to_segment(p, s):
    """Get the distance between 1 point and 1 segment."""
    area = triangle_area([p, s.p1, s.p2])
    d = get_distance_point_to_point(s.p1, s.p2)
    h = 2. * area / d

    return h


def get_slope(p1, p2):
    """Get the slope of the line going through 2 points."""
    x1, y1 = p1.get_coordinate()
    x2, y2 = p2.get_coordinate()
    slope = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else None

    return slope


def get_intercept(p1, p2):
    """Get the intercept of the line going through 2 points."""
    x1, y1 = p1.get_coordinate()
    x2, y2 = p2.get_coordinate()
    intercept = (x2 * y1 - x1 * y2) / (x2 - x1) if x2 - x1 != 0 else None

    return intercept


def get_line(p1, p2):
    """Get the slope and intercept of the line going through 2 points."""
    slope = get_slope(p1, p2)
    intercept = get_intercept(p1, p2)

    return slope, intercept


def cross_product(v1, v2):
    """Calculate the cross product of 2 vectors as (x1 * y2 - x2 * y1)."""

    return v1.x * v2.y - v2.x * v1.y


def inner_product(v1, v2):
    """Calculate the inner product of 2 vectors as (x1 * x2 + y1 * y2)."""

    return v1.x * v2.x + v1.y * v2.y


def triangle_area(points):
    """Calculate the area of a triangle."""
    a = get_distance_point_to_point(points[0], points[1])
    b = get_distance_point_to_point(points[1], points[2])
    c = get_distance_point_to_point(points[2], points[0])
    p = (a + b + c) / 2.
    s = (p * (p - a) * (p - b) * (p - c)) ** 0.5

    return s
