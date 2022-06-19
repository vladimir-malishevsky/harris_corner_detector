import cv2
import numpy as np
from numpy.random import randint
import skimage

WINDOW_SIZE = 21
K = 0.04
SIGMA = 0.2
THRESHOLD = 0.2
NMS_MIN_DISTANCE = 5

FILTER_X = np.array(
    (
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ), dtype="int32"
)

FILTER_Y = np.array(
    (
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ), dtype="int32"
)

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

MARKER_COLOR = (0, 0, 255)  # Колір маркерів (червоний)
MARKER_THICKNESS = 1  # Товщина маркерів

TEXT_COLOR = (0, 255, 0)  # Колір тексту (червоний)
TEXT_THICKNESS = 2  # Товщина тексту
TEXT_SCALE = 1  # розмір тексту


def harris_detector(img: np.ndarray, window_size: int = WINDOW_SIZE,
                    k: int = K, sigma: float = SIGMA,
                    threshold: int = None, nms_min_distance: int = NMS_MIN_DISTANCE) -> [np.ndarray]:
    """
    Функція пошуку точок інтересу (кутів)
    Arguments:
        :param img: зображення I
        :param window_size: розмір вікна w(y, x)
        :param k: емпіричний коефіцієнт
        :param sigma: сигма для пошуку похідних Ix, Iy
        :param threshold: коефіцієнт порогової фільтрації
        :param nms_min_distance: дистанція між максимумами
    Returns:
        corners (np.ndarray): массив з координатами точок інтересу
    """

    # переводимо зображення у відтінки сірого
    grayscale = bgr2gray(img.copy())

    # розмиваємо зображення
    img_gaussian = cv2.GaussianBlur(grayscale, (window_size, window_size), sigma)

    # шукаємо похідні
    Ix = convolve(img_gaussian, FILTER_X)  # Похідна по X
    Iy = convolve(img_gaussian, FILTER_Y)  # Похідна по Y

    # розраховуємо квадрати похідних та перехресну фільтрацію
    Ixx = np.square(Ix)
    Iyy = np.square(Iy)
    Ixy = Ix * Iy

    # обробка гаусом
    Sxx = convolve(Ixx, GAUSS)
    Syy = convolve(Iyy, GAUSS)
    Sxy = convolve(Ixy, GAUSS)

    # шукаємо відгуки Харріса
    trace = Sxx + Syy
    determinant = Sxx * Syy - np.square(Sxy)
    harris = determinant - k * np.square(trace)

    # відсікаємо значення менше нуля
    harris = np.maximum(harris, 0)

    # нормалізуємо в межах від 0 до 1
    harris = cv2.normalize(harris, harris, 0, 1, cv2.NORM_MINMAX)

    # придушуємо немаксимуми і робимо порогову фільтрацію
    threshold = threshold if threshold else np.max(harris) * 0.01
    corners = skimage.feature.peak_local_max(
        harris,
        min_distance=nms_min_distance,
        threshold_abs=threshold
    )

    return corners, grayscale  # повертаємо координати точок інтересу


def show_matches(img1: np.ndarray, img2: np.ndarray, matches: list):
    """
    Функція, яка виводить два зображення поруч та робить зіставлення точок інтересу
    Arguments:
        :param img1: перше зображення
        :param img2: друге зображення
        :param matches: список координат точок для першого і другого зображення
    """
    offset = img1.shape[1]  # відступ від лівого зображення
    color = _random_color()  # генеруємо випадковий початковий колір

    # конкатинуємо два зображення по осі X після чого отримаємо матрицю поєднаних двох зображень
    concat = np.concatenate((img1, img2), axis=1)

    for [y1, x1], [y2, x2] in matches:  # проходимось (ітеруємось) по координатах
        draw_marker(concat, (x1, y1), color, 2)  # малюємо маркер на першому зображення
        draw_marker(concat, (x2 + offset, y2), color, 2)  # малюємо маркер на другому зображенні
        cv2.line(concat, (x1, y1), (x2 + offset, y2), color, 2)  # малюємо лінію зіставлення між маркерами
        color = _random_color()  # генеруємо новий випадковий колір

    draw_text(concat, (50, 50), f'Matches: {len(matches)}')  # виводимо текст з інформацією про зіставлені точки
    cv2.imshow('Matching', concat)  # виводимо зображення у формі


def match_descriptors(desctriptors1, desctriptors2):
    """
    Зіставлення точок інтересу за патч дескрипторами
    Arguments:
        :param desctriptors1: список патч-тескрипторів для першого зображення
        :param desctriptors2: список патч-тескрипторів для другого зображення
    Returns:
        matches (list): список координат зіставлених точок інтересу для двох зображень
    """
    matches = []  # список координат зіставлених точок

    for desc1, desc1_y, desc1_x in desctriptors1:  # проходимось (ітеруємось) по дескрипторам для першого зображення
        distances = []  # список евклідових відстаней від дескриптора першого зображення до дескрипторів другого зображення
        for desc2, desc2_y, desc2_x in desctriptors2:  # проходимось (ітеруємось) по дескрипторам для другого зображення
            #  шукаємо евклідову відстань
            square = np.square(desc1 - desc2)
            sum_square = np.sum(square)
            distance = np.sqrt(sum_square)
            #  додаємо евклідову відстань в список
            distances.append([distance, desc2_y, desc2_x])

        # перевіряємо чи наш список дистанцій не пустий, в інакшому випадку переходимо до наступної ітерації
        if not distances:
            continue

        distances.sort(key=lambda l: l[0])  # сортуємо список відстаней від найменшої до найбільшої

        last_item = distances[0]  # отримуємо найменшу відстань
        before_last_item = distances[1]  # отримуємо перед найменшу відстань
        result = last_item[0] / before_last_item[0]  # ділимо найменшу на перед найменшу відстань

        if result < 0.8:  # якщо вираз вірний то ми можемо припустити, що точки за патч-дескрипторами однакові
            matches.append([[desc1_y, desc1_x], [last_item[1], last_item[2]]])  # додаємо координати точок у список

    return matches  # повертаємо список координат точок які були успішно зіставлені для обох зображень


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
    # Якщо ми будемо шукати дескриптор для точки яка знаходиться на межах зображення,
    # то ми можемо вийти за його межі, через що отримаємо виключення, тому підемо цікавим шляхом,
    # а саме розширимо наше зображення з кожної з сторін на половину розміру нашого вікна (window_size // 2)
    pad = window_size // 2  # розраховуємо, на скільки ми будемо розширювати матрицю зображення

    p_gray = np.pad(grayscale, (pad,), 'constant')  # розширюємо зображення та заповнюємо 0

    descriptors = []  # массив з десткрипторами

    num_neighbor = pad  # вказуємо число сусідніх комірок, також розраховується з половини від вікна

    for y, x in corners:  # ітеруємось по y та x координатах точок інтересу
        left = pad + x - num_neighbor  # шукаємо лівого сусіда
        right = pad + x + num_neighbor + 1  # шукаємо правого сусіда
        top = pad + y + num_neighbor + 1  # шукаємо верхнього сусіда
        bottom = pad + y - num_neighbor  # шукаємо нижнього сусіда

        neighbors = p_gray[bottom:top, left:right]  # отримуємо матрицю сусідів у вигляді списку
        descriptor = np.array(neighbors).ravel()  # переводимо матрицю у вигляді списку у матрицю numpy та робимо вектором
        descriptors.append([descriptor, y, x])  # додаємо патч-дескриптор та координати точки інтересу у список

    return descriptors  # повертаємо список патч-дескрипторів з координатами точок інтересу


def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Функція згортки
    Arguments:
        :param img: забраження
        :param kernel: ядро
    Returns:
        g (np.ndarray): результат згортки
    """
    # Щоб не вийти за межі зображення при згортці, ми його розширюємо на половину без остачі від ширини вікна
    img_height = img.shape[0]  # отримуємо висоту зображення
    img_width = img.shape[1]  # Отримуємо ширину зображення
    pad_height = kernel.shape[0] // 2  # розраховуємо висоту розширення
    pad_width = kernel.shape[1] // 2  # розраховуємо ширину розширення

    pad = ((pad_height, pad_height), (pad_height, pad_width))  # розраховуємо розширення
    g = np.empty(img.shape, dtype=np.float64)
    img = np.pad(img, pad, mode='constant', constant_values=0)
    # робимо згортку
    for y in np.arange(pad_height, img_height + pad_height):  # проходимось (ітеруємось) по Y
        for x in np.arange(pad_width, img_width + pad_width):  # проходимось (ітеруємось) по X
            roi = img[y - pad_height:y + pad_height + 1, x - pad_width:x + pad_width + 1]  # шукаємо roi
            g[y - pad_height, x - pad_width] = (roi * kernel).sum()  # записуємо результат

    if (g.dtype == np.float64):  # якщо це ядро гауса
        kernel = kernel / 255.0
        kernel = (kernel * 255).astype(np.uint8)
    else:  # інакше
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
    # height, width, _ = img.shape
    # for y in range(height):  # проходимось по осі Y
    #     for x in range(width):  # проходимось по осі X
    #         b, g, r = img[y, x]  # за допомогою деструктуризації отримуємо яскравість кольорів пікселя
    #         Y = b * 0.0722 + g * 0.7152 + r * 0.2126
    #         img[y, x] = Y  # вносимо зміни у піксель
    #
    # return img  # повертаємо оброблене зображення
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # повертаємо оброблене зображення


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
    for y, x in corners:  # проходимось (ітеруємось) по y, x
        draw_marker(img, (x, y), color, thickness)  # малюємо маркер за координатами
    return img  # повертаємо результат


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
        img (np.ndarray): зображення з нанесиним маркером
    """
    x, y = pos  # за допомогою деструктуризації отримуємо x, y
    # малюємо маркер за допомогою двух ліній
    cv2.line(img, (x - 4, y), (x + 4, y), color, thickness)  # малюємо лінію за координатами
    cv2.line(img, (x, y - 4), (x, y + 4), color, thickness)  # малюємо лінію за координатами
    return img  # повертаємо результат


def draw_text(img: np.ndarray, pos: tuple, text: str,
              color: tuple = TEXT_COLOR, thickness: int = TEXT_THICKNESS, scale: int = TEXT_SCALE) -> np.ndarray:
    """
    Малює текст за координатами
    Arguments:
        :param img: забраження
        :param text: текст
        :param pos: координати
        :param color: колір тексту
        :param thickness: товщина тексту
        :param scale: розмір тексту
    Returns:
        img (np.ndarray): зображення з нанесиним текстом
    """
    # виводимо текст за координатами
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return img  # повертаємо результат


def _random_color() -> tuple:
    """
    Генератор випадкових кольорів
    Returns:
        color (tuple): колір у форматі RGB
    """
    return COLORS[randint(0, len(COLORS) - 1)]  # повертаємо випадковий колір
