import numpy as np
import cv2
import kernel as k

def normalize(out):
    cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    return out


img = cv2.imread('Lena.jpg')
cv2.imshow("Original", img)
img = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Grayscale", img)
sigma = 0.7

kernel = k.gaussian(sigma)
img_conv = k.convolution(img, kernel)



Gx, Gy = k.sobel(img_conv)
Gx = k.convolution(Gx, kernel)
Gy = k.convolution(Gy, kernel)


mag = np.sqrt(Gx ** 2 + Gy ** 2)
angle = np.arctan2(Gy, Gx) * 180 / np.pi

cv2.imshow("Gradient Intensity", normalize(mag))
# cv2.imshow("angel", normalize(angle))

nomaxsup = k.non_max_suppression(mag, angle)

cv2.imshow("non maximum suppression", normalize(nomaxsup))

# t = k.globalThresholding(nomaxsup)

res, weak, strong = k.threshold(nomaxsup)
cv2.imshow("double thresholding", normalize(res))
out = k.hysteresis(res, weak, strong)
cv2.imshow("hysteresis", normalize(out))

cv2.waitKey(0)
cv2.destroyAllWindows()