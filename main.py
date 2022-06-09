import cv2
import numpy as np

# from own_cv2 import harris_detector
import own_cv2 as own

# original_img = cv2.imread('patterns/Image.jpg')
# # other_img = cv2.imread('patterns/Image_1.jpg')
# other_img = cv2.imread('patterns/Image.jpg')

# original_img = cv2.imread('patterns/Image.jpg')
# other_img = cv2.imread('images/Image.jpg')

original_img = cv2.imread('images/Image.jpg')
other_img = cv2.imread('images/Image_1.jpg')


grayscale_original_img = own.grayscale(original_img)
grayscale_other_img = own.grayscale(other_img)

original_img_corners, harris1 = own.harris_detector(grayscale_original_img)
other_img_corners, harris2 = own.harris_detector(grayscale_other_img)

descriptors_original_image = own.patch_descriptors(harris1, original_img_corners)
descriptors_other_image = own.patch_descriptors(harris2, other_img_corners)

own.match_descriptors(descriptors_original_image, descriptors_other_image)


cv2.imshow('Original', own.add_markers(original_img, original_img_corners))
cv2.imshow('Other', own.add_markers(other_img, other_img_corners))
cv2.waitKey()







# harris_detector('images/corners.png')
# harris_detector('images/Image.jpg')
# harris_detector('patterns/Image.jpg')


