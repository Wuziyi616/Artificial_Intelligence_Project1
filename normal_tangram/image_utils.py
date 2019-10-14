"""This file contains functions for processing image"""

import cv2
import math
import copy
import numpy as np
import matplotlib.pyplot as plt


def binarize_image(image):
    """Binarize image pixel values to 0 and 255."""
    unique_values = np.unique(image)
    if len(unique_values) == 2:
        if (unique_values == np.array([0., 255.])).all():
            return image
    mean = image.mean()
    image[image > mean] = 255
    image[image <= mean] = 0

    return image


def read_gray_image(path):
    """Read in a gray scale image."""
    image = cv2.imread(path, 0)

    return image


def read_image(path):
    """Read in a RGB image."""
    image = cv2.imread(path)

    return image


def save_image(image, path):
    """Save image using cv2."""
    cv2.imwrite(path, image)


def get_black_area(raw_mask):
    """Get the area of black values which needs to be filled with Tangram.
    Input:
        raw_mask: input image of the Tangram problem, np.array with black area == 0
    Return:
        black: area of black values
    """
    h, w = raw_mask.shape
    black = h * w - np.count_nonzero(raw_mask)

    return black


def get_unit_length(raw_mask, standard_s=8.):
    """Get the unit length for a Tangram problem.
    For example, if an input mask has an area == 32, while a typical Tangram has an area == 8,
        then the unit length will be 2 for this problem.
    Input:
        raw_mask: input image of the Tangram problem, np.array with black area == 0
        standard_s: standard square of a set of Tangram, typically 8
    Return:
        unit_length: the length in the mask that equals to 1 in a typical Tangram
    """
    black_area = get_black_area(raw_mask)
    unit_length = math.sqrt(float(black_area) / float(standard_s))

    return unit_length


def show_gray_image(image):
    """Show gray scale image."""
    plt.imshow(image, cmap='gray')
    plt.show()


def show_image(image):
    """Show RGB image."""
    plt.imshow(image)
    plt.show()


def draw_polygon(image, polygon, color=128):
    """Draw polygon on an image.
    Can be used to transfer state when you fill an element on the mask.
    """
    img = copy.deepcopy(image)
    points = []
    for point in polygon.points:
        points.append(point.get_coordinate())
    points = np.round(np.array([points])).astype(np.int32)
    points[:, :, :] = points[:, :, [1, 0]]

    cv2.fillPoly(img, points, color)

    return img


def show_polygon(image, polygon):
    """Show a polygon on a gray scale image."""
    img = copy.deepcopy(image)
    threshold = 128
    img[image > threshold] = 255
    img[image <= threshold] = 0
    img = draw_polygon(img, polygon, color=128)
    show_gray_image(img)


def get_final_result(raw_mask, elements, colors):
    """Draw elements on raw_mask and return the final solutions."""
    assert len(elements) == len(colors)
    img = copy.deepcopy(raw_mask)
    if len(img.shape) == 2:
        # extend it to RGB form image
        img = np.stack([img, img, img], axis=-1)
    for i in range(len(elements)):
        img = draw_polygon(img, polygon=elements[i], color=colors[i])

    return img


def detect_corners(image, threshold):
    """Detect corners in a given image using contour method.
    Note that the image should be GRAY form!
    The biggest advantage of this method is that, it can output corners in order (clockwise)!
    Return:
        corners: a list of vertices that are divided by different connected area
    """
    # need to reverse the pixel values first since black area == 0 and white area == 255
    # and we only want to contours of black pixels
    img = copy.deepcopy(image)
    img[image > 128] = 0
    img[image <= 128] = 255
    _, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # len(contours) == number of connected areas in the img
    # put them into different lists because we need to generate segments from them separately later
    # contours: [connected_area_num, num, 1, (x, y)]
    connected_area_num = len(contours)
    all_corners = []

    for area_idx in range(connected_area_num):
        num = len(contours[connected_area_num - area_idx - 1])
        loc = np.zeros((2, num), dtype=np.int32)
        # reverse the x, y coordinate to match image axis
        # also reverse the order of points detected because
        # the original output from cv2.findContours is antiClockwise
        for i in range(num):
            loc[[1, 0], i] = contours[connected_area_num - area_idx - 1][num - i - 1][0]
        corners = refine_vertex(loc, threshold)
        all_corners.append(corners)

    return all_corners


def refine_vertex(loc, threshold):
    """Refine the corners detected in a mask.
    Main purpose is to specify ONE coordinate per vertex.
    Then the returned points will be used as the basis of the task.
    Input:
        loc: candidate vertex points, ([x1, x2, ...], [y1, y2, ...]), [2, num]
        threshold: min distance between 2 vertices, typically 0.2 * unit_length
    Return:
        vertices: a list of points that each is one vertex, [[x1, y1], [x2, y2], ...]
    """
    vertices = []  # [[[x11, y11], [x12, y12], ...], [[x21, y21], [x22, y22], ...], ...]
    flag = False
    for i in range(len(loc[0])):
        x, y = loc[0][i], loc[1][i]
        for j in range(len(vertices)):
            mean_vertex = np.mean(vertices[j], axis=0)
            if math.sqrt((mean_vertex[0] - x) ** 2 + (mean_vertex[1] - y) ** 2) < threshold:
                vertices[j].append([x, y])
                flag = True
                break
        if flag:
            flag = False
            continue
        vertices.append([[x, y]])
    vertices = [np.round(np.mean(vertices[i], axis=0)).astype(np.int32) for i in range(len(vertices))]

    return vertices


def point_is_inside_mask(mask, p, threshold, value=0):
    """Judge whether a point is inside an area that has the given pixel VALUES.
    But we need to have some error threshold for non-integer coordinate issue.
    Note that mask should be in gray form.
    """
    h, w = mask.shape
    x, y = p.get_coordinate()
    x = int(x)
    y = int(y)
    threshold = int(threshold)

    if x + threshold < 0 or x - threshold >= h or \
            y + threshold < 0 or y - threshold >= w:
        # out of the mask range
        return False

    x_min = max(0, x - threshold)
    x_max = min(h, x + threshold)
    y_min = max(0, y - threshold)
    y_max = min(w, y + threshold)

    if (mask[x_min:x_max, y_min:y_max] == value).any():
        return True
    return False
