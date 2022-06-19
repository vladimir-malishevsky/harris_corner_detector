import cv2
import numpy as np
import scipy as sp
from scipy import signal as signal
from scipy.ndimage import gaussian_filter
import skimage

import matplotlib.pyplot as plt
from numpy.random import randint

WINDOW_SIZE = 21
K = 0.04
SIGMA = 0.2
THRESHOLD = 0.2
NMS_MIN_DISTANCE = 5

MARKER_COLOR = (0, 0, 255)  # Колір маркерів (червоний)
MARKER_THICKNESS = 1  # Товщина маркерів

# Sobel x-axis kernel
FILTER_X = np.array(
    (
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ), dtype="int32"
)

# Sobel y-axis kernel
FILTER_Y = np.array(
    (
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
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

COLORS = [
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


def harris_detector(img: np.ndarray, window_size: int = WINDOW_SIZE,
                    k: int = K, sigma1: float = SIGMA,
                    threshold: int = None, nms_min_distance: int = NMS_MIN_DISTANCE) -> np.ndarray:
    """
    Функція пошуку точок інтересу (кутів)
    Arguments:
        :param img: зображення I
        :param window_size: розмір вікна w(y, x)
        :param k: емпіричний коефіцієнт
        :param sigma1: сигма для пошуку похідних Ix, Iy
        :param sigma2: сигма для пошуку Sxx, Syy, Sxy
        :param threshold: коефіцієнт порогової фільтрації
        :param nms_min_distance: дистанція між максимумами
    Returns:
        corners (np.ndarray): массив з координатами точок інтересу
    """

    print('Переводимо зображення У відтінки сірого...')

    grayscale = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    img_gaussian = cv2.GaussianBlur(grayscale, (window_size, window_size), sigma1)

    print('Пошук похідних...')
    Ix = convolve(img_gaussian, FILTER_X)  # Похідна по X
    Iy = convolve(img_gaussian, FILTER_Y)  # Похідна по Y

    # dx = signal.convolve2d(img_gaussian, [-1, 0, 1], mode='same')
    # dy = signal.convolve2d(img_gaussian, [[-1], [0], [1]], mode='same')

    # Ix = gaussian_filter(grayscale, order=[0, 1], sigma=sigma1, truncate=trunc1)
    # Iy = gaussian_filter(grayscale, order=[1, 0], sigma=sigma2, truncate=trunc1)

    # Ix = signal.convolve2d(img_gaussian, [[-1, 0, 1]], mode='same')
    # Iy = signal.convolve2d(img_gaussian, [[-1], [0], [1]], mode='same')

    print('Квадрат похідних...')
    # Квадрат похідних
    Ixx = np.square(Ix)
    Iyy = np.square(Iy)
    Ixy = Ix * Iy  # перехресна фільтрація
    print('Обробка гаусом...')
    # g_dx2 = convolve(dx2, GAUSS)
    # g_dy2 = convolve(dy2, GAUSS)
    # g_dxy = convolve(dxy, GAUSS)
    # g_dx2 = cv2.GaussianBlur(dx2, (5, 5), 1)
    # g_dy2 = cv2.GaussianBlur(dy2, (5, 5), 1)
    # g_dxy = cv2.GaussianBlur(dxy, (5, 5), 1)
    #
    # Sxx = gaussian_filter(Ixx, 0.6)
    # Syy = gaussian_filter(Iyy, 0.6)
    # Sxy = gaussian_filter(Ixy, 0.6)
    Sxx = convolve(Ixx, GAUSS)
    Syy = convolve(Iyy, GAUSS)
    Sxy = convolve(Ixy, GAUSS)

    # Sxx = gaussian_filter(Ixx, sigma=sigma2, truncate=trunc2)
    # Syy = gaussian_filter(Iyy, sigma=sigma2, truncate=trunc2)
    # Sxy = gaussian_filter(Ixy, sigma=sigma2, truncate=trunc2)

    # Sxx = signal.convolve2d(grayscale, h, mode='same')
    # Syy = signal.convolve2d(grayscale, h, mode='same')
    # Sxy = signal.convolve2d(grayscale, h, mode='same')

    print('Шукаємо кути...')
    # Детектор Харріса r(harris) = det - k*(trace**2)
    trace = Sxx + Syy
    determinant = Sxx * Syy - np.square(Sxy)
    harris = determinant - k * np.square(trace)

    print('Відсікаємо від`ємні значення...')
    harris = np.maximum(harris, 0)  # замінюємо x <= 0 на 0

    print('Нормалізуємо...')
    harris = cv2.normalize(harris, harris, 0, 1, cv2.NORM_MINMAX)  # нормалізуємо матрицю (0-1)

    print('Придушуємо не максимуми...')
    threshold = threshold if threshold else np.max(harris) * 0.01
    corners = skimage.feature.peak_local_max(
        harris,
        min_distance=nms_min_distance,
        threshold_abs=threshold
    )

    return corners


def draw_matches(img1, img2, matches):
    offset = img1.shape[1]
    color = _random_color()

    concat = np.concatenate((img1, img2), axis=1)

    for [y1, x1], [y2, x2] in matches:
        draw_marker(concat, (x1, y1), color, 2)
        draw_marker(concat, (x2 + offset, y2), color, 2)
        cv2.line(concat, (x1, y1), (x2 + offset, y2), color, 2)
        color = _random_color()

    cv2.putText(concat, f'Matches: {len(matches)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('', concat)


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

        if not distances:
            continue

        distances.sort(key=lambda l: l[0])

        last_item = distances[0]
        before_last_item = distances[1]
        result = last_item[0] / before_last_item[0]

        if result < 0.8:
            matches.append([[desc1_y, desc1_x], [last_item[1], last_item[2]]])

    return matches


def patch_descriptors(grayscale: np.ndarray, corners: np.ndarray, window_size=WINDOW_SIZE):
    """
    Пошук патч дескрипторів для точок інтересу
    Arguments:
        :param grayscale: зображення у відтінках сірого
        :param corners: координати точок інтересу
        :param window_size: розмір вікна
    Returns:
        descriptors (list): список патч дескрипторів та їх координати
    """

    pad = window_size // 2

    p_gray = np.pad(grayscale, (pad,), 'constant')  # розширюємо зображення та заповнюємо його 0

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

    return descriptors


def convolve(img: np.ndarray, kernel):
    """
    Функція згортки
    Arguments:
        :param img: забраження
        :param kernel: ядро
    Returns:
        returns some poop
    """
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


def bgr2gray(img: np.ndarray) -> np.ndarray:
    """
    Функція конвертації BGR зображення у відтінки сірого
    Arguments:
        :param img: зображення
    Returns:
        img (np.ndarray): зображення у відтінках сірого
    """
    height, width, lol = img.shape
    for y in range(height):  # проходимось по осі Y
        for x in range(width):  # проходимось по осі X
            b, g, r = img[y, x]  # за допомогою деструктуризації отримуємо яскравість кольорів пікселя
            Y = b * 0.0722 + g * 0.7152 + r * 0.2126
            img[y, x] = [Y, Y, Y]  # вносимо зміни у піксель
    return img  # повертаємо оброблене зображення
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def draw_markers(img: np.ndarray, corners: np.ndarray,
                 color: tuple = MARKER_COLOR, thickness: int = MARKER_THICKNESS) -> np.ndarray:
    """
    Малює маркери за координатами
    Arguments:
        :param img: забраження
        :param corners: массив координат точок інтересу
        :param color: колір маркеру
        :param thickness: товщина маркеру
    Returns:
        img (np.ndarray): зображення з нанесиними маркерами
    """
    for y, x in corners:
        draw_marker(img, (x, y), color, thickness)
    return img


def draw_marker(img: np.ndarray, pos: tuple,
                color: tuple = MARKER_COLOR, thickness: int = MARKER_THICKNESS) -> np.ndarray:
    """
    Малює маркер за координатами
    Arguments:
        :param img: забраження
        :param pos: координати точоки інтересу
        :param color: колір маркеру
        :param thickness: товщина маркеру
    Returns:
        img (np.ndarray): зображення з нанесиними маркерами
    """
    x, y = pos
    cv2.line(img, (x - 4, y), (x + 4, y), color, thickness)
    cv2.line(img, (x, y - 4), (x, y + 4), color, thickness)
    return img

def _random_color() -> tuple:
    """
    Генератор випадкових кольорів
    Returns:
        color (tuple): колір у форматі RGB
    """
    return COLORS[randint(0, len(COLORS) - 1)]
