import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage
from skimage import feature


def my_harris(img_dir, window_size, k, threshold):
    img = cv2.imread(img_dir)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gaussian = cv2.GaussianBlur(gray, (5, 5), 0.5)
    print(img_gaussian)

    height = img.shape[0]  # .shape[0] outputs height
    width = img.shape[1]  # .shape[1] outputs width .shape[2] outputs color channels of image

    #   Step 1 - Calculate the x e y image derivatives (dx e dy)
    dx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=window_size)
    dy = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=window_size)

    #   Step 2 - Calculate product and second derivatives (dx2, dy2 e dxy)
    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxy = dx * dy

    #   Step 3 - Calcular a soma dos produtos das derivadas para cada pixel (Sx2, Sy2 e Sxy)
    print("Finding Corners...")
    Sx2 = cv2.GaussianBlur(dx2, (5, 5), 1)
    Sy2 = cv2.GaussianBlur(dy2, (5, 5), 1)
    Sxy = cv2.GaussianBlur(dxy, (5, 5), 1)

    trace = Sx2 + Sy2
    determinant = Sx2 * Sy2 - Sxy ** 2
    matrix_R = determinant - k * (trace ** 2)

    print("Apply a threshold...")
    #   Step 6 - Apply a threshold
    cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)
    matrix_R = feature.peak_local_max(matrix_R, min_distance=10, threshold_abs=threshold)

    for y, x in matrix_R:
        cv2.line(img, (x - 4, y), (x + 4, y), (0, 0, 255), 1)
        cv2.line(img, (x, y - 4), (x, y + 4), (0, 0, 255), 1)

    pad = window_size // 2
    tmp = np.pad(gray, (6,), 'symmetric')
    for x, y in matrix_R:
        lol = np.array([
            [tmp[x - 1 + pad, y - 1 + pad], tmp[x + pad, y + pad], tmp[x + 1 + pad, y - 1 + pad]],
            [tmp[x - 1 + pad, y     + pad], tmp[x + pad, y + pad], tmp[x + 1 + pad, y     + pad]],
            [tmp[x - 1 + pad, y + 1 + pad], tmp[x + pad, y + pad], tmp[x + 1 + pad, y + 1 + pad]]
        ])


    cv2.imshow('', img)
    cv2.imwrite('../images/gg.jpg', img)
    cv2.waitKey()






my_harris("../images/corners.png", 5, 0.04, 0.30)
# my_harris("image.jpg", 5, 0.04, 0.30)

