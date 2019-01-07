import cv2
import matplotlib.pyplot as plt
import numpy as np


image_id = "0000003"
orig_image_path = "./data/%s_orig.png" % image_id
blur_image_path = "./data/%s_blur.png" % image_id
kernel_image_path = "./data/%s_psf.png" % image_id

orig_image = cv2.imread(orig_image_path, -1)
blur_image = cv2.imread(blur_image_path, -1)
kernel_image = cv2.imread(kernel_image_path, -1)


def motion_blur(orig_image, kernel_image):
    kernel_image = kernel_image.astype(np.float32)
    kernel_image /= np.sum(kernel_image)
    return cv2.filter2D(orig_image, -1, kernel_image, borderType=cv2.BORDER_REPLICATE)


add_blur = motion_blur(orig_image, kernel_image)
plt.subplot(2, 2, 1)
plt.imshow(orig_image)
plt.subplot(2, 2, 2)
plt.imshow(blur_image)
plt.subplot(2, 2, 3)
plt.imshow(kernel_image)
plt.subplot(2, 2, 4)
plt.imshow(add_blur)
plt.show()
