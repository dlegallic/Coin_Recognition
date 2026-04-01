import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


#Implement Thresholds to test if better !


#Naive prewitt operator implementation
def prewitt_operatorSLOW(imageArray):
    #Kernel used to approximate the derivatives
    Mx = [[-1, 0, 1],
          [-1, 0, 1],
          [-1, 0, 1]]
    
    My = [[-1,-1,-1],
          [ 0, 0, 0],
          [ 1, 1, 1]]

    #Setup the output array
    height, width = imageArray.shape
    prewittImg = np.zeros((height, width), dtype=int)
    
    #Manual convolution (extremely inefficient)
    for i in range(1,height-1):
        for j in range(1, width-1):
            Gx = (Mx[0][0] * imageArray[i + 1][j + 1]) + \
                 (Mx[0][1] * imageArray[i + 1][j]) + \
                 (Mx[0][2] * imageArray[i + 1][j - 1]) + \
                 (Mx[1][0] * imageArray[i][j + 1]) + \
                 (Mx[1][1] * imageArray[i][j]) + \
                 (Mx[1][2] * imageArray[i][j - 1]) + \
                 (Mx[2][0] * imageArray[i - 1][j + 1]) + \
                 (Mx[2][1] * imageArray[i - 1][j]) + \
                 (Mx[2][2] * imageArray[i - 1][j - 1])

            Gy = (My[0][0] * imageArray[i + 1][j + 1]) + \
                 (My[0][1] * imageArray[i + 1][j]) + \
                 (My[0][2] * imageArray[i + 1][j - 1]) + \
                 (My[1][0] * imageArray[i][j + 1]) + \
                 (My[1][1] * imageArray[i][j]) + \
                 (My[1][2] * imageArray[i][j - 1]) + \
                 (My[2][0] * imageArray[i - 1][j + 1]) + \
                 (My[2][1] * imageArray[i - 1][j]) + \
                 (My[2][2] * imageArray[i - 1][j - 1])

            normalized = np.sqrt(Gx**2 + Gy**2)
            prewittImg[i][j] = normalized
            
    return prewittImg

#Naive sobel operator implementation
def sobel_operatorSLOW(imageArray):
    #Another kernel used to approximate the derivatives
    Mx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
    
    My = [[-1,-2,-1],
          [ 0, 0, 0],
          [ 1, 2, 1]]
    
    #Initialize the output array
    height, width = imageArray.shape
    sobelImg = np.zeros((height, width), dtype=int)
    
    #Manual convolution
    for i in range(1,height-1):
        for j in range(1, width-1):
            Gx = (Mx[0][0] * imageArray[i + 1][j + 1]) + \
                 (Mx[0][1] * imageArray[i + 1][j]) + \
                 (Mx[0][2] * imageArray[i + 1][j - 1]) + \
                 (Mx[1][0] * imageArray[i][j + 1]) + \
                 (Mx[1][1] * imageArray[i][j]) + \
                 (Mx[1][2] * imageArray[i][j - 1]) + \
                 (Mx[2][0] * imageArray[i - 1][j + 1]) + \
                 (Mx[2][1] * imageArray[i - 1][j]) + \
                 (Mx[2][2] * imageArray[i - 1][j - 1])

            Gy = (My[0][0] * imageArray[i + 1][j + 1]) + \
                 (My[0][1] * imageArray[i + 1][j]) + \
                 (My[0][2] * imageArray[i + 1][j - 1]) + \
                 (My[1][0] * imageArray[i][j + 1]) + \
                 (My[1][1] * imageArray[i][j]) + \
                 (My[1][2] * imageArray[i][j - 1]) + \
                 (My[2][0] * imageArray[i - 1][j + 1]) + \
                 (My[2][1] * imageArray[i - 1][j]) + \
                 (My[2][2] * imageArray[i - 1][j - 1])

            normalized = np.sqrt(Gx**2 + Gy**2)
            sobelImg[i][j] = normalized
            
    return sobelImg


#Vectorized and efficient prewitt implementation
def prewittVectorized(imageArray):
    Mx = [[-1, 0, 1],
          [-1, 0, 1],
          [-1, 0, 1]]
    
    My = [[-1,-1,-1],
          [ 0, 0, 0],
          [ 1, 1, 1]]

    height, width = imageArray.shape
    
    Gx = signal.convolve2d(imageArray, Mx, mode='valid')
    Gy = signal.convolve2d(imageArray, My, mode='valid')
    grad = np.sqrt(Gx**2 + Gy**2)
    return grad

def gaussianBlur(imageArray):
    B=(1/159)*np.array([[2,  4,  5,  4, 2],
               [4,  9, 12,  9, 4],
               [5, 12, 15, 12, 5],
               [4,  9, 12,  9, 4],
               [2,  4,  5,  4, 2]])
    
    #Apply gaussian blur
    filteredImg = signal.convolve2d(imageArray, B, mode='valid')
    return filteredImg

#Canny filter implementation, mostly optimized
def cannyFilter(imageArray, function):
    #Apply gaussian blur
    filteredImg= gaussianBlur(imageArray)

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
    Gx = signal.convolve2d(filteredImg, Mx, mode='valid')
    Gy = signal.convolve2d(filteredImg, My, mode='valid')
    grad = np.sqrt(Gx**2 + Gy**2)
    
    #Change the direction to the closest one (0°, 45°, 90° or 135°)
    direction = (np.rad2deg(np.arctan2(Gy, Gx)) + 180) % 180
    direction = (np.round(direction / 45) * 45) % 180
    
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
    
    grad = grad / grad.max() * 255
    
    #Double thresholding and hysteresis
    highThreshold = np.percentile(grad, 98) 
    lowThreshold  = 0.6*highThreshold
    # print(highThreshold)
    # plt.hist(grad.ravel(), bins=256, range=[1,256])
    # plt.axvline(highThreshold, color='orange', linestyle='--', label='highT')
    # plt.axvline(lowThreshold, color='orange', linestyle='--', label='lowT')
    # plt.show()
    edges = np.zeros_like(grad)
    edges[grad >= highThreshold] = 200
    edges[grad > lowThreshold] += 55
    return edges
