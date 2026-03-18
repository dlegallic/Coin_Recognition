import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def prewitt_operatorSLOW(image):
    Mx = [[-1, 0, 1],
          [-1, 0, 1],
          [-1, 0, 1]]
    
    My = [[-1,-1,-1],
          [ 0, 0, 0],
          [ 1, 1, 1]]
    
    img = plt.imread(image)
    gray_img = np.round(0.299 * img[:, :, 0] +
                        0.587 * img[:, :, 1] +
                        0.114 * img[:, :, 2]).astype(np.uint8)
    
    height, width = gray_img.shape
    prewittImg = np.zeros((height, width), dtype=int)
    
    for i in range(1,height-1):
        for j in range(1, width-1):
            Gx = (Mx[0][0] * gray_img[i + 1][j + 1]) + \
                 (Mx[0][1] * gray_img[i + 1][j]) + \
                 (Mx[0][2] * gray_img[i + 1][j - 1]) + \
                 (Mx[1][0] * gray_img[i][j + 1]) + \
                 (Mx[1][1] * gray_img[i][j]) + \
                 (Mx[1][2] * gray_img[i][j - 1]) + \
                 (Mx[2][0] * gray_img[i - 1][j + 1]) + \
                 (Mx[2][1] * gray_img[i - 1][j]) + \
                 (Mx[2][2] * gray_img[i - 1][j - 1])

            Gy = (My[0][0] * gray_img[i + 1][j + 1]) + \
                 (My[0][1] * gray_img[i + 1][j]) + \
                 (My[0][2] * gray_img[i + 1][j - 1]) + \
                 (My[1][0] * gray_img[i][j + 1]) + \
                 (My[1][1] * gray_img[i][j]) + \
                 (My[1][2] * gray_img[i][j - 1]) + \
                 (My[2][0] * gray_img[i - 1][j + 1]) + \
                 (My[2][1] * gray_img[i - 1][j]) + \
                 (My[2][2] * gray_img[i - 1][j - 1])

            normalized = np.sqrt(Gx**2 + Gy**2)
            prewittImg[i][j] = normalized
            
    return prewittImg

def sobel_operatorSLOW(image):
    Mx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
    
    My = [[-1,-2,-1],
          [ 0, 0, 0],
          [ 1, 2, 1]]
    
    img = plt.imread(image)
    gray_img = np.round(0.299 * img[:, :, 0] +
                        0.587 * img[:, :, 1] +
                        0.114 * img[:, :, 2]).astype(np.uint8)
    
    height, width = gray_img.shape
    sobelImg = np.zeros((height, width), dtype=int)
    
    
    
    for i in range(1,height-1):
        for j in range(1, width-1):
            Gx = (Mx[0][0] * gray_img[i + 1][j + 1]) + \
                 (Mx[0][1] * gray_img[i + 1][j]) + \
                 (Mx[0][2] * gray_img[i + 1][j - 1]) + \
                 (Mx[1][0] * gray_img[i][j + 1]) + \
                 (Mx[1][1] * gray_img[i][j]) + \
                 (Mx[1][2] * gray_img[i][j - 1]) + \
                 (Mx[2][0] * gray_img[i - 1][j + 1]) + \
                 (Mx[2][1] * gray_img[i - 1][j]) + \
                 (Mx[2][2] * gray_img[i - 1][j - 1])

            Gy = (My[0][0] * gray_img[i + 1][j + 1]) + \
                 (My[0][1] * gray_img[i + 1][j]) + \
                 (My[0][2] * gray_img[i + 1][j - 1]) + \
                 (My[1][0] * gray_img[i][j + 1]) + \
                 (My[1][1] * gray_img[i][j]) + \
                 (My[1][2] * gray_img[i][j - 1]) + \
                 (My[2][0] * gray_img[i - 1][j + 1]) + \
                 (My[2][1] * gray_img[i - 1][j]) + \
                 (My[2][2] * gray_img[i - 1][j - 1])

            normalized = np.sqrt(Gx**2 + Gy**2)
            sobelImg[i][j] = normalized
            
    return sobelImg

def prewittVectorized(image):
    Mx = [[-1, 0, 1],
          [-1, 0, 1],
          [-1, 0, 1]]
    
    My = [[-1,-1,-1],
          [ 0, 0, 0],
          [ 1, 1, 1]]
    
    img = plt.imread(image)
    gray_img = np.round(0.299 * img[:, :, 0] +
                        0.587 * img[:, :, 1] +
                        0.114 * img[:, :, 2]).astype(np.uint8)
    
    height, width = gray_img.shape
    
    Gx = signal.convolve2d(gray_img, Mx, mode='same', boundary='fill')
    Gy = signal.convolve2d(gray_img, My, mode='same', boundary='fill')
    grad = np.sqrt(Gx**2 + Gy**2)
    return grad

def prewittVectorized(image, seuil):
    Mx = [[-1, 0, 1],
          [-1, 0, 1],
          [-1, 0, 1]]
    
    My = [[-1,-1,-1],
          [ 0, 0, 0],
          [ 1, 1, 1]]
    
    img = plt.imread(image)
    gray_img = np.round(0.299 * img[:, :, 0] +
                        0.587 * img[:, :, 1] +
                        0.114 * img[:, :, 2]).astype(np.uint8)
    
    height, width = gray_img.shape
    
    Gx = signal.convolve2d(gray_img, Mx, mode='same', boundary='fill')
    Gy = signal.convolve2d(gray_img, My, mode='same', boundary='fill')
    grad = np.sqrt(Gx**2 + Gy**2)
    grad = np.where(grad > seuil, 255, 0)
    return grad
    

plt.figure()
#plt.imshow(seuil("euros.png", 100), cmap='gray')
plt.imsave('euros.png', prewittVectorized("euros.jpeg", 70), cmap='gray', format='png')
plt.show()    
    
