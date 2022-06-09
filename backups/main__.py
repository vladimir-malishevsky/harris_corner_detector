import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage
from skimage import feature


def my_harris(img_dir, window_size, k, threshold):
    img = cv2.imread(img_dir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

    height = img.shape[0]  # .shape[0] outputs height
    width = img.shape[1]  # .shape[1] outputs width .shape[2] outputs color channels of image

    #   Step 1 - Calculate the x e y image derivatives (dx e dy)
    dx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=5)
    # dy, dx = np.gradient(gray)
    #   Step 2 - Calculate product and second derivatives (dx2, dy2 e dxy)
    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxy = dx * dy

    offset = int(window_size / 2)
    #   Step 3 - Calcular a soma dos produtos das derivadas para cada pixel (Sx2, Sy2 e Sxy)
    print("Finding Corners...")
    Sx2 = cv2.GaussianBlur(dx2, (5, 5), 0)
    Sy2 = cv2.GaussianBlur(dy2, (5, 5), 0)
    Sxy = cv2.GaussianBlur(dxy, (5, 5), 0)
    trace = Sx2 + Sy2
    determinant = Sx2 * Sy2 - Sxy ** 2
    matrix_R = determinant - k * (trace ** 2)
    # for i in range(len(Sxy)):
    #     det = np.linalg.det(H[..., i])
    #     tr = np.matrix.trace(H)
    #     R = det - k * (tr ** 2)
    #     temp.append(R)
    # matrix_R = np.reshape(np.array(temp), np.shape(matrix_R))
    # print(matrix_R)

    # for y in range(offset, height - offset):
    #     for x in range(offset, width - offset):
    #         Sx2 = np.sum(dx2[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
    #         Sy2 = np.sum(dy2[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
    #         Sxy = np.sum(dxy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
    #
    #         #   Step 4 - Define the matrix H(x,y)=[[Sx2,Sxy],[Sxy,Sy2]]
    #         H = np.array([[Sx2, Sxy], [Sxy, Sy2]])
    #
    #         #   Step 5 - Calculate the response function ( R=det(H)-k(Trace(H))^2 )
    #         det = np.linalg.det(H)
    #         tr = np.matrix.trace(H)
    #         R = det - k * (tr ** 2)
    #         matrix_R[y - offset, x - offset] = R
    print("Apply a threshold...")
    #   Step 6 - Apply a threshold
    cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)

    matrix_R = feature.peak_local_max(matrix_R, min_distance=10, threshold_abs=threshold)

    for y, x in matrix_R:
        # cv2.line(img, (x - 4, y), (x + 4, y), (0, 0, 255), 1)
        # cv2.line(img, (x, y - 4), (x, y + 4), (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 255, 0))
    # for y in range(offset, height - offset):
    #     for x in range(offset, width - offset):
    #         value = matrix_R[y, x]
    #         if value > threshold:
    #             # cornerList.append([x, y, value])
    #             # cv2.circle(img, (x, y), 3, (0, 255, 0))
    #             cv2.line(img, (x - 4, y), (x + 4, y), (0, 0, 255), 1)
    #             cv2.line(img, (x, y - 4), (x, y + 4), (0, 0, 255), 1)



    # cv2.imwrite("%s_threshold_%s.png"%(img_dir[5:-4],threshold), img)
    cv2.imshow('', img)
    cv2.imwrite('../images/gg.jpg', img)
    cv2.waitKey()






my_harris("../images/corners.png", 5, 0.04, 0.30)
# my_harris("image.jpg", 5, 0.04, 0.30)
# ttt = cv2.imread('image.jpg')
# cv2.line(ttt, (10, 10), (20,20), (255, 0, 0), 3)
# cv2.imshow('', ttt)
# cv2.waitKey()