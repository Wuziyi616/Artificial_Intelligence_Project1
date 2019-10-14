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


def resize_image_to_unit_length(image, unit_length, standard_s=8., fill_ratio=1.):
    """Resize input image to a given unit_length, for example 10.
    Because I think when handling any input image Tangram problem,
        it's impossible to detect and use corners and segments of raw_mask.
    So I've to iterate over all pixels within the image to place elements.
    In order to reduce the searching complexity, I think we should resize image to a smaller degree.
    Keep original image ratio!
    """
    ori_unit_length = get_unit_length(image, standard_s) * math.sqrt(fill_ratio)
    resize_ratio = float(unit_length) / float(ori_unit_length)
    h, w = image.shape
    new_size = (int(round(w * resize_ratio)), int(round(h * resize_ratio)))
    # image is binary so I think we should use nearest
    new_image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)

    return new_image


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


def compute_iou(mask, solutions):
    """Compute IOU between two masks.
    The area we want has value == 0.
    Input:
        mask: [height, width]
        solutions: [num, height, width]
    """
    gt = copy.deepcopy(mask)
    pred = copy.deepcopy(solutions)
    gt[mask > 128] = 0
    gt[mask <= 128] = 1
    pred[solutions > 128] = 0
    pred[solutions <= 128] = 1

    # intersection and union
    gt = gt.reshape((-1,)).astype(np.int32)  # [h * w]
    pred = pred.reshape((solutions.shape[0], -1)).astype(np.int32)  # [num, h * w]
    area1 = np.sum(gt)
    area2 = np.sum(pred, axis=1)  # [num]
    intersection = np.sum(gt[None, :] * pred, axis=1)  # [num]
    union = area1 + area2 - intersection
    ious = intersection / (union + 1e-6)

    return ious


def draw_polygons_on_mask(mask, elements, color=0):
    img = copy.deepcopy(mask)
    for element in elements:
        img = draw_polygon(img, element['element'], color)

    return img
