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


def get_unit_length(raw_mask, standard_s=64.):
    """Get the unit length for a Tangram problem.
    For example, if an input mask has an area == 64, while a typical 13 Tangram has an area == 64,
        then the unit length will be 1 for this problem.
    Input:
        raw_mask: input image of the Tangram problem, np.array with black area == 0
        standard_s: standard square of a set of 13 Tangram, typically 64
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


def get_final_result(grid, elements, colors):
    """Draw elements on grid and returns the final solution."""
    img = copy.deepcopy(grid)
    if len(img.shape) == 2:
        # extend it to RGB form image
        img = np.stack([img, img, img], axis=-1)
    for i in range(len(elements)):
        for j in range(elements[i].area):
            img[elements[i].coordinates[j][0], elements[i].coordinates[j][1]] = colors[i]

    return img


def segment_image(image, tangram_s):
    """Since we know all elements in a 13 Tangram can be decomposed into small 1x1 squares,
    I want to segment the original image into grid form that,
    each pixel corresponds to one 1x1 square.
    """
    # get unit_length
    unit_length = int(round(get_unit_length(image, tangram_s)))

    # first reverse image to set black area == 1 and white area == 0
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[image > 128] = 0
    mask[image <= 128] = 1
    w_sum = np.sum(mask, axis=0)
    h_sum = np.sum(mask, axis=1)
    loc1 = np.where(h_sum >= unit_length * 0.5)
    start_x = loc1[0][0]
    end_x = loc1[0][-1] + 1
    loc2 = np.where(w_sum >= unit_length * 0.5)
    start_y = loc2[0][0]
    end_y = loc2[0][-1] + 1
    h = end_x - start_x
    w = end_y - start_y
    assert (h % unit_length == 0 and w % unit_length == 0)

    # pad image
    ori_h, ori_w = mask.shape
    new_h = (ori_h // unit_length + 2) * unit_length
    new_w = (ori_w // unit_length + 2) * unit_length
    new_image = np.ones((new_h, new_w), dtype=np.uint8) * 255
    pad_x_start = unit_length - (start_x % unit_length)
    pad_y_start = unit_length - (start_y % unit_length)

    new_image[pad_x_start:pad_x_start + ori_h, pad_y_start:pad_y_start + ori_w] = image

    # generate grid
    h = new_h // unit_length
    w = new_w // unit_length
    grid = np.ones((h, w), dtype=np.uint8) * 255

    # iterate over small squares and compare areas
    mask = np.zeros_like(new_image, dtype=np.uint8)
    mask[new_image > 128] = 0
    mask[new_image <= 128] = 1
    for i in range(h):
        for j in range(w):
            area = \
                np.sum(mask[unit_length * i:unit_length * (i + 1), unit_length * j:unit_length * (j + 1)])
            if area > (unit_length ** 2) * 0.5:
                grid[i, j] = 0

    return unit_length, new_image, grid
