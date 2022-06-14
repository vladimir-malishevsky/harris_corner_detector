import cv2
import numpy as np
import skimage

import matplotlib.pyplot as plt
from numpy.random import randint

default_window_size = 5
default_k = 0.04
default_threshold = 0.2

# Sobel x-axis kernel
SOBEL_X = np.array(
    (
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ), dtype="int32"
)

# Sobel y-axis kernel
SOBEL_Y = np.array(
    (
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ), dtype="int32"
)

# Gaussian kernel
GAUSS = np.array(
    (
        [1 / 16, 2 / 16, 1 / 16],
        [2 / 16, 4 / 16, 2 / 16],
        [1 / 16, 2 / 16, 1 / 16]
    ), dtype="float64"
)

colors = [
    (255, 255, 255),  # White
    (255, 0, 0),  # Red
    (0, 255, 0),  # Lime
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray
    (128, 0, 0),  # Maroon
    (128, 128, 0),  # Olive
    (0, 128, 0),  # Green
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (0, 0, 128)  # Navy
]


def harris_detector(gray, window_size=default_window_size, k=default_k, threshold=default_threshold,
                    nms_min_distance=5):
    """
    My own harris detector function
    Arguments:
        :param gray: image
        :param window_size: window size image
        :param k: empirical coefficient
        :param threshold: the value of the constraints for threshold filtering angles
        :param nms_min_distance:
    Returns:
        returns some poop
    """
    if window_size % 2 != 1:
        raise ValueError("Розмір вікна має бути непарним!")
    # print(f'{image_path}: Переводимо зображення У відтінки сірого...')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Переводимо зображення У відтінки сірого
    #
    img_gaussian = cv2.GaussianBlur(gray, (window_size, window_size), 0.5)  # Згорткою шукаємо Ix, Iy через DoG
    print('Пошук похідних...')
    dx = convolve(img_gaussian, SOBEL_X)  # Похідна по X
    dy = convolve(img_gaussian, SOBEL_Y)  # Похідна по Y
    print('Квадрат похідних...')
    # Квадрат похідних
    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxy = dx * dy  # перехресна фільтрація
    print('Обробка гаусом...')
    # g_dx2 = convolve(dx2, GAUSS)
    # g_dy2 = convolve(dy2, GAUSS)
    # g_dxy = convolve(dxy, GAUSS)
    g_dx2 = cv2.GaussianBlur(dx2, (5, 5), 1)
    g_dy2 = cv2.GaussianBlur(dy2, (5, 5), 1)
    g_dxy = cv2.GaussianBlur(dxy, (5, 5), 1)

    print('Шукаємо кути...')
    # Детектор Харріса r(harris) = det - k*(trace**2)
    trace = g_dx2 + g_dy2
    determinant = g_dx2 * g_dy2 - np.square(g_dxy)
    harris = determinant - k * np.square(trace)


    print('Відсікаємо від`ємні значення...')
    harris = np.maximum(harris, 0)  # замінюємо x <= 0 на 0

    print('Нормалізуємо...')
    harris = cv2.normalize(harris, harris, 0, 1, cv2.NORM_MINMAX)  # нормалізуємо матрицю (0-1)

    print('Придушуємо не максимуми...')
    corners = skimage.feature.peak_local_max(harris, min_distance=nms_min_distance, threshold_abs=np.max(harris) * 0.01)

    return corners


# def drawMatches(img1, img2, matches):
def draw_matches(img1, img2, matches):
    offset = img1.shape[1]
    concat = np.concatenate((img1, img2), axis=1)

    color = _random_color()

    for [y1, x1], [y2, x2] in matches:
        cv2.line(concat, (x1 - 4, y1), (x1 + 4, y1), color, 2)
        cv2.line(concat, (x1, y1 - 4), (x1, y1 + 4), color, 2)

        cv2.line(concat, (x2 - 4 + offset, y2), (x2 + 4 + offset, y2), color, 2)
        cv2.line(concat, (x2 + offset, y2 - 4), (x2 + offset, y2 + 4), color, 2)

        cv2.line(concat, (x1, y1), (x2 + offset, y2), color, 2)
        color = _random_color()

    cv2.imshow('', concat)


# def drawMatches_offset(img1, img2, matches):
#     offset_x = img1.shape[1]
#
#     prepared_img1 = np.concatenate((img1, np.full_like(img2, 0)))
#     prepared_img2 = np.concatenate((np.full_like(img1, 0), img2))
#     cv2.imshow('', np.concatenate((prepared_img1, prepared_img2), axis=1))
#     cv2.waitKey()
#     exit()
#     prepared_img1 = np.pad()
#     concat = np.concatenate((img1, img2), axis=1)
#
#     color = _random_color()
#
#     for [y1, x1], [y2, x2] in matches:
#         cv2.line(concat, (x1 - 4, y1), (x1 + 4, y1), color, 2)
#         cv2.line(concat, (x1, y1 - 4), (x1, y1 + 4), color, 2)
#
#         cv2.line(concat, (x2 - 4 + offset_x, y2), (x2 + 4 + offset_x, y2), color, 2)
#         cv2.line(concat, (x2 + offset_x, y2 - 4), (x2 + offset_x, y2 + 4), color, 2)
#
#         cv2.line(concat, (x1, y1), (x2 + offset_x, y2), color, 2)
#         color = _random_color()
#
#     cv2.imshow('', concat)
#     cv2.waitKey()



def match_descriptors(desctriptors1, desctriptors2):
    """
    Пошук патч дескрипторів
    Arguments:
        :param desctriptors1:
        :param desctriptors2:
    Returns:
        returns some poop
    """
    matches = []

    for desc1, desc1_y, desc1_x in desctriptors1:
        distances = []
        desc2_y, desc2_x = 0, 0
        for desc2, desc2_y, desc2_x in desctriptors2:
            square = np.square(desc1 - desc2)
            sum_square = np.sum(square)
            distance = np.sqrt(sum_square)
            distances.append([distance, desc2_y, desc2_x])
            # distances.append(distance)
        # print(f'Y: {desc1_y}; X: {desc1_x};')
        # print(distances)
        distances.sort(key=lambda l: l[0])
        # distances.sort()

        last_item = distances[0]
        before_last_item = distances[1]
        result = last_item[0] / before_last_item[0]

        if result < 0.8:
            # matches.append([desc1_y, desc1_x])
            # matches.append([last_item[1], last_item[2]])
            matches.append([[desc1_y, desc1_x], [last_item[1], last_item[2]]])

        # print(f'Y: {desc1_y}; X: {desc1_x};')
        # if result < 0.8:
        #     print(f'True')
        # else:
        #     print(f'False')
    return matches


def patch_descriptors(gray, corners, window_size=default_window_size):
    """
    Пошук патч дескрипторів
    Arguments:
        :param gray:
        :param corners:
        :param window_size:
    Returns:
        returns some poop
    """

    g_img_height = gray.shape[0]
    g_img_width = gray.shape[1]
    pad = window_size // 2

    p_gray = np.pad(gray, (pad,), 'constant')  # розширюємо зображення та заповнюємо його 0

    descriptors = []

    num_neighbor = pad

    for y, x in corners:
        left = pad + x - num_neighbor
        right = pad + x + num_neighbor + 1
        top = pad + y + num_neighbor + 1
        bottom = pad + y - num_neighbor

        neighbors = p_gray[bottom:top, left:right]
        descriptor = np.array(neighbors).ravel()
        descriptors.append([descriptor, y, x])

        # xy = num_neighbor + 1
        # raw = p_gray[bottom:top, left:right]
        # cv2.line(raw, (xy - 4, xy), (xy + 4, xy), (0, 0, 255), 1)
        # cv2.line(raw, (xy, xy - 4), (xy, xy + 4), (0, 0, 255), 1)
        # cv2.imshow('', raw)
        # cv2.waitKey()
    return descriptors


def add_markers(img, corners):
    for y, x in corners:
        cv2.line(img, (x - 4, y), (x + 4, y), (0, 0, 255), 1)
        cv2.line(img, (x, y - 4), (x, y + 4), (0, 0, 255), 1)
    return img


def convolve(img, kernel):
    """
    Функція згортки
    Arguments:
        :param img: забраження
        :param kernel: ядро
    Returns:
        returns some poop
    """
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError("Підтримуються тільки непарні числа у фільтрі")

    img_height = img.shape[0]  # отримуємо висоту зображення
    img_width = img.shape[1]  # Отримуємо ширину зображення
    pad_height = kernel.shape[0] // 2  # розраховуємо висоту здвигу
    pad_width = kernel.shape[1] // 2  # розраховуємо ширину здвигу

    pad = ((pad_height, pad_height), (pad_height, pad_width))  # розраховуємо здвиг
    g = np.empty(img.shape, dtype=np.float64)
    img = np.pad(img, pad, mode='constant', constant_values=0)
    # робимо згортку
    for i in np.arange(pad_height, img_height + pad_height):
        for j in np.arange(pad_width, img_width + pad_width):
            roi = img[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
            g[i - pad_height, j - pad_width] = (roi * kernel).sum()

    if (g.dtype == np.float64):
        kernel = kernel / 255.0
        kernel = (kernel * 255).astype(np.uint8)
    else:
        g = g + abs(np.amin(g))
        g = g / np.amax(g)
        g = (g * 255.0)
    return g


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _color_generator():
    for color in colors:
        yield color


def _random_color():
    return colors[randint(0, len(colors) - 1)]
# def harris_detector(image_path, window_size=5, k=0.03, threshold=0.2):
#     """
#     My own harris detector function
#     Arguments:
#         :param image_path: path to image
#         :param window_size: window size image
#         :param k: empirical coefficient
#         :param threshold: the value of the constraints for threshold filtering angles
#     Returns:
#         returns some poop
#     """
#
#     img = cv2.imread(image_path)  # Зчитаємо зображення
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Переводимо зображення У відтінки сірого
#
#     img_gaussian = cv2.GaussianBlur(gray, (5, 5), 0.5)  # Згорткою шукаємо Ix, Iy через DoG
#
#     #   Step 1 - Calculate the x e y image derivatives (dx e dy)
#     # dx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=window_size)
#     # dy = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=window_size)
#
#     # Horizontal Derivative
#     dx = sp.signal.convolve2d(img_gaussian, [[1, 0, -1]], mode='same')
#
#     # Vertical Derivative
#     dy = sp.signal.convolve2d(img_gaussian, [[1], [0], [-1]], mode='same')
#
#     #   Step 2 - Calculate product and second derivatives (dx2, dy2 e dxy)
#     dx2 = np.square(dx)
#     dy2 = np.square(dy)
#     dxy = dx * dy
#
#     sx2 = cv2.GaussianBlur(dx2, (5, 5), 1)
#     sy2 = cv2.GaussianBlur(dy2, (5, 5), 1)
#     sxy = cv2.GaussianBlur(dxy, (5, 5), 1)
#
#     trace = sx2 + sy2
#     determinant = sx2 * sy2 - sxy ** 2
#     matrix_r = determinant - k * (trace ** 2)
#
#     cv2.normalize(matrix_r, matrix_r, 0, 1, cv2.NORM_MINMAX)  # нормалізуємо матрицю
#     matrix_r = skimage.feature.peak_local_max(matrix_r, min_distance=10, threshold_abs=threshold)
#
#     # for x in range(matrix_r.shape[0]):
#     #     for y in range(matrix_r.shape[1]):
#     #         cv2.line(img, (x - 4, y), (x + 4, y), (0, 0, 255), 1)
#     #         cv2.line(img, (x, y - 4), (x, y + 4), (0, 0, 255), 1)
#     for y, x in matrix_r:
#         cv2.line(img, (x - 4, y), (x + 4, y), (0, 0, 255), 1)
#         cv2.line(img, (x, y - 4), (x, y + 4), (0, 0, 255), 1)
#
#     # pad = window_size // 2
#     # tmp = np.pad(gray, (6,), 'symmetric')
#     # for x, y in matrix_r:
#     #     lol = np.array([
#     #         [tmp[x - 1 + pad, y - 1 + pad], tmp[x + pad, y + pad], tmp[x + 1 + pad, y - 1 + pad]],
#     #         [tmp[x - 1 + pad, y + pad], tmp[x + pad, y + pad], tmp[x + 1 + pad, y + pad]],
#     #         [tmp[x - 1 + pad, y + 1 + pad], tmp[x + pad, y + pad], tmp[x + 1 + pad, y + 1 + pad]]
#     #     ])
#
#     cv2.imwrite('gg.jpg', img)
#     # cv2.imshow('', img)
#     # cv2.waitKey()
#
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("harris detector")
#     plt.waitforbuttonpress()
