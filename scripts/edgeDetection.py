import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#Naive prewitt operator implementation
def prewitt_operatorSLOW(image):
    #Kernel used to approximate the derivatives
    Mx = [[-1, 0, 1],
          [-1, 0, 1],
          [-1, 0, 1]]
    
    My = [[-1,-1,-1],
          [ 0, 0, 0],
          [ 1, 1, 1]]
    
    #Load the image and convert it to a grayscale
    img = plt.imread(image)
    gray_img = np.round(0.299 * img[:, :, 0] +
                        0.587 * img[:, :, 1] +
                        0.114 * img[:, :, 2]).astype(np.uint8)
    
    #Setup the output array
    height, width = gray_img.shape
    prewittImg = np.zeros((height, width), dtype=int)
    
    #Manual convolution (extremely inefficient)
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

#Naive sobel operator implementation
def sobel_operatorSLOW(image):
    #Another kernel used to approximate the derivatives
    Mx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
    
    My = [[-1,-2,-1],
          [ 0, 0, 0],
          [ 1, 2, 1]]
    
    #Load the image and convert it to grayscale
    img = plt.imread(image)
    gray_img = np.round(0.299 * img[:, :, 0] +
                        0.587 * img[:, :, 1] +
                        0.114 * img[:, :, 2]).astype(np.uint8)
    
    #Initialize the output array
    height, width = gray_img.shape
    sobelImg = np.zeros((height, width), dtype=int)
    
    #Manual convolution
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


#Vectorized and efficient prewitt implementation
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

#Canny filter implementation, mostly optimized
def cannyFilter(image, function):
    #Load and convert to a gray image
    img = plt.imread(image)
    grayImg = np.round(0.299 * img[:, :, 0] +
                        0.587 * img[:, :, 1] +
                        0.114 * img[:, :, 2]).astype(np.uint8)
    B=(1/159)*np.array([[2,  4,  5,  4, 2],
               [4,  9, 12,  9, 4],
               [5, 12, 15, 12, 5],
               [4,  9, 12,  9, 4],
               [2,  4,  5,  4, 2]])
    
    #Apply gaussian blur
    filteredImg = signal.convolve2d(grayImg, B)
    
    #Select a type of edge detection
    if(function=='sobel'):
        Mx = [[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]
        My = [[-1,-2,-1],
              [ 0, 0, 0],
              [ 1, 2, 1]]
    elif(function=='prewitt'):
        Mx = [[-1, 0, 1],
              [-1, 0, 1],
              [-1, 0, 1]]
        My = [[-1,-1,-1],
              [ 0, 0, 0],
              [ 1, 1, 1]]
   
    #Calculate the gradient and direction of the edge
    Gx = signal.convolve2d(filteredImg, Mx, mode='same', boundary='fill')
    Gy = signal.convolve2d(filteredImg, My, mode='same', boundary='fill')
    grad = np.sqrt(Gx**2 + Gy**2)
    
    #Change the direction to the closest one (0°, 45°, 90° or 135°)
    direction = (np.rad2deg(np.arctan2(Gy, Gx)) + 180) % 180
    direction_q = (np.round(direction / 45) * 45)   
    
    #Gradient non-maximum suppression (edge thinning)
    points = np.nonzero(grad)
    for i, j in zip(*points):
        if i == 0 or i == grad.shape[0]-1 or j == 0 or j == grad.shape[1]-1:
            continue
        
        match direction[i, j]:
            case 0:
                if grad[i, j] < grad[i, j-1] or grad[i, j] < grad[i, j+1]:
                    grad[i,j]=0
            case 45:
                if grad[i, j] < grad[i+1, j-1] or grad[i, j] < grad[i-1, j+1]:
                    grad[i,j]=0
            case 90:
                if grad[i, j] < grad[i-1, j] or grad[i, j] < grad[i+1, j]:
                    grad[i,j]=0
            case 135:
                if grad[i, j] < grad[i+1, j+1] or grad[i, j] < grad[i-1, j-1]:
                    grad[i,j]=0
                    
    #Double thresholding and hysteresis
    highThreshold = np.percentile(grad, 99) 
    lowThreshold  = np.percentile(grad, 98)
    
    edges = np.zeros_like(grad)
    edges[grad >= highThreshold] = 200
    edges[grad > lowThreshold] += 55
    return edges

plt.imsave('../testImage/cannyEuros.png', cannyFilter('../testImage/moreEuros.jpeg', 'prewitt'), cmap='gray')
plt.show()
    
    
