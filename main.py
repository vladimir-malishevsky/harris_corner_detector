import cv2
import numpy as np

# from own_cv2 import harris_detector
import own_cv2 as own

# original_img = cv2.imread('patterns/Image1.jpg')
# other_img = cv2.imread('patterns/Image2.jpg')


# original_img = cv2.imread('patterns/Image.jpg')
# other_img = cv2.imread('images/Image.jpg')

original_img = cv2.imread('desk/Image-00.jpg')
other_img = cv2.imread('desk/Image-01.jpg')

original_img_corners = own.harris_detector(original_img)
other_img_corners = own.harris_detector(other_img)

descriptors_original_image = own.patch_descriptors(cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY), original_img_corners)
descriptors_other_image = own.patch_descriptors(cv2.cvtColor(other_img, cv2.COLOR_BGR2GRAY), other_img_corners)

matches = own.match_descriptors(descriptors_original_image, descriptors_other_image)

own.draw_matches(original_img, other_img, matches)


# print(matches)
cv2.imshow(f'Corners1', own.draw_markers(original_img, original_img_corners))
cv2.imshow(f'Corners2', own.draw_markers(other_img, other_img_corners))
cv2.waitKey()


# original_with_markers = own.add_markers(original_img, matches[0])
# other_with_markers = own.add_markers(other_img, matches[1])
#
# cv2.imshow(f'Original: {original_with_markers.shape[0]}x{original_with_markers.shape[1]}', original_with_markers)
# cv2.imshow(f'Other: {other_with_markers.shape[0]}x{other_with_markers.shape[1]}', other_with_markers)
# cv2.waitKey()

# cv2.imshow('Original', own.add_markers(original_img, original_img_corners))
# cv2.imshow('Other', own.add_markers(other_img, other_img_corners))
# cv2.waitKey()



# harris_detector('images/corners.png')
# harris_detector('images/Image.jpg')
# harris_detector('patterns/Image.jpg')


