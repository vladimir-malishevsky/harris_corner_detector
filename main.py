import cv2
import own_cv2 as own

# Відкриваємо перше та друге зображення
original_img = cv2.imread('desk/Image-00.jpg')
other_img = cv2.imread('desk/Image-01.jpg')

# шукаємо кути для першого та другого зображення
original_img_corners, original_img_grayscale = own.harris_detector(original_img)  # шукаємо кути для першого зображення
other_img_corners, other_img_grayscale = own.harris_detector(other_img)  # шукаємо кути для другого зображення

# шукаємо патч-дескриптори для першого та другого зображення
descriptors_original_image = own.patch_descriptors(original_img_grayscale, original_img_corners)
descriptors_other_image = own.patch_descriptors(other_img_grayscale, other_img_corners)

# шукаємо координати точок інтересу які співпали
matches = own.match_descriptors(descriptors_original_image, descriptors_other_image)

# візуалізуємо зіставлення
own.show_matches(original_img, other_img, matches)

# візуалізуємо знайдені кути для обох зображень
cv2.imshow(f'Corners1', own.draw_markers(original_img, original_img_corners))
cv2.imshow(f'Corners2', own.draw_markers(other_img, other_img_corners))
cv2.waitKey()


