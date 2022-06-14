import cv2
import numpy as np

# from own_cv2 import harris_detector
import own_cv2 as own

# original_img = cv2.imread('patterns/Image.jpg')
# # other_img = cv2.imread('patterns/Image_1.jpg')
# other_img = cv2.imread('patterns/Image.jpg')

# original_img = cv2.imread('patterns/Image.jpg')
# other_img = cv2.imread('images/Image.jpg')

original_img = cv2.imread('desk/Image-00.jpg')
other_img = cv2.imread('desk/Image-01.jpg')


grayscale_original_img = own.grayscale(original_img)
grayscale_other_img = own.grayscale(other_img)

original_img_corners = own.harris_detector(grayscale_original_img, nms_min_distance=5)
other_img_corners = own.harris_detector(grayscale_other_img, nms_min_distance=5)

descriptors_original_image = own.patch_descriptors(grayscale_original_img, original_img_corners)
descriptors_other_image = own.patch_descriptors(grayscale_other_img, other_img_corners)

matches = own.match_descriptors(descriptors_original_image, descriptors_other_image)

own.drawMatches(original_img, other_img, matches)

print(matches)
# cv2.imshow(f'Corners', own.add_markers(original_img, original_img_corners))
# cv2.imshow(f'Corners', cv2.drawMatches(original_img, matches[0], original_img, matches[1]))


original_with_markers = own.add_markers(original_img, matches[0])
other_with_markers = own.add_markers(other_img, matches[1])

cv2.imshow(f'Original: {original_with_markers.shape[0]}x{original_with_markers.shape[1]}', original_with_markers)
cv2.imshow(f'Other: {other_with_markers.shape[0]}x{other_with_markers.shape[1]}', other_with_markers)
cv2.waitKey()

# cv2.imshow('Original', own.add_markers(original_img, original_img_corners))
# cv2.imshow('Other', own.add_markers(other_img, other_img_corners))
# cv2.waitKey()



# harris_detector('images/corners.png')
# harris_detector('images/Image.jpg')
# harris_detector('patterns/Image.jpg')


