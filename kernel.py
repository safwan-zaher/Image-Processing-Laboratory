import numpy as np
import math
import cv2



def convolution(img, kernel):
    n = (kernel.shape[0] // 2)
    r = img.shape[0]
    c = img.shape[1]
    out = np.zeros((r, c))
    img = cv2.copyMakeBorder(src=img, top=n, bottom=n,left=n,right=n,borderType=cv2.BORDER_CONSTANT)

    for i in range(n, r - n):
        for j in range(n, c - n):
            res = 0
            for x in range(-n, n):
                for y in range(-n, n):
                    res += kernel[x + n, y + n] * img.item(i - x, j - y)
            out[i, j] = res
    out = out[n: -n, n: -n]
    return out


def gaussian(sigma=0.7):
    # bitwise operation will make it odd always (5,7)
    n = int(sigma * 7) | 1
    const = 1 / (2.0 * 3.1416 * sigma ** 2)
    kernel = np.zeros((n,n))
    n = n // 2
    sum = 0
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            res1 = -(i * i + j * j) / (2.0 * sigma ** 2)
            res2 = const * math.exp(res1)
            kernel[i + n][j + n] = res2
            sum += res2
    return kernel / sum


def deriv_x(kernel, sigma):
    n = kernel.shape[0] // 2
    r = kernel.shape[0]
    c = kernel.shape[1]
    new = np.zeros((r, c))
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            res = -(kernel[i + n, j + n] * i) / (sigma ** 2)
            new[i + n, j + n] = res
    print(new)
    return new


def deriv_y(kernel, sigma):
    n = kernel.shape[0] // 2
    r = kernel.shape[0]
    c = kernel.shape[1]
    new = np.zeros((r, c))
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            res = -(kernel[i + n, j + n] * j) / (sigma ** 2)
            new[i + n, j + n] = res
    print(new)
    return new



def sobel(img):
    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return (Gx, Gy)


def threshold(img, low = 0.05, high = 0.09):
    highThreshold = img.max() * high
    lowThreshold = highThreshold * low

    r = img.shape[0]
    c = img.shape[1]
    new = np.zeros((r, c))

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    new[strong_i, strong_j] = strong
    new[weak_i, weak_j] = weak

    return (new, weak, strong)


def hysteresis(image, weak, strong=255):
    M, N = image.shape
    out = image.copy()

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (image[i, j] == weak):
                if np.any(image[i - 1:i + 2, j - 1:j + 2] == strong):
                    out[i, j] = strong
                else:
                    out[i, j] = 0
    return out

def non_max_suppression(img, angle):
    image = img.copy()
    image = image / image.max() * 255
    out = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            q = 0
            r = 0
            if (-22.5 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180) or (-180 <= angle[i, j] <= -157.5):
                r = image[i, j - 1]
                q = image[i, j + 1]
            elif (-67.5 <= angle[i, j] <= -22.5) or (112.5 <= angle[i, j] <= 157.5):
                r = image[i - 1, j + 1]
                q = image[i + 1, j - 1]
            elif (67.5 <= angle[i, j] <= 112.5) or (-112.5 <= angle[i, j] <= -67.5):
                r = image[i - 1, j]
                q = image[i + 1, j]
            elif (22.5 <= angle[i, j] < 67.5) or (-167.5 <= angle[i, j] <= -112.5):
                r = image[i + 1, j + 1]
                q = image[i - 1, j - 1]


            if (image[i, j] >= q) and (image[i, j] >= r):
                out[i, j] = image[i, j]
            else:
                out[i, j] = 0

    return out





