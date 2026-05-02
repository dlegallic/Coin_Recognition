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
    
    Gx = signal.convolve2d(imageArray, Mx, mode='same', boundary='symm')
    Gy = signal.convolve2d(imageArray, My, mode='same', boundary='symm')
    grad = np.sqrt(Gx**2 + Gy**2)
    return grad

#return the hysteresis of a given edge array by applying a recursive algorithm (SLOW)
def hysteresis(edges, weak, strong):
    h, w = edges.shape
    results = edges.copy()
    #recursive algorithm
    stabilized = False
    while not stabilized:
        stabilized = True
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if results[i, j] == weak:
                    if (results[i-1:i+2, j-1:j+2] == strong).any():
                        results[i, j] = strong
                        stabilized = False
                    else:
                        results[i, j] = 0
    return results

#Gaussian Blur, sigma=1.4
def gaussianBlur(imageArray):
    B=(1/159)*np.array([[2,  4,  5,  4, 2],
               [4,  9, 12,  9, 4],
               [5, 12, 15, 12, 5],
               [4,  9, 12,  9, 4],
               [2,  4,  5,  4, 2]], dtype=np.float32)
    
    #Apply gaussian blur
    filteredImg = signal.convolve2d(imageArray, B, mode='same', boundary='symm')
    return filteredImg

#Canny filter implementation, mostly optimized
def cannyFilter(imageArray, function='sobel', nth=99):
    #Apply gaussian blur
    filteredImg= gaussianBlur(imageArray)

    #Select a type of edge detection
    if(function=='sobel'):
        Mx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        My = np.array([[-1,-2,-1],
                       [ 0, 0, 0],
                       [ 1, 2, 1]])
    elif(function=='prewitt'):
        Mx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]])
        My = np.array([[-1,-1,-1],
                       [ 0, 0, 0],
                       [ 1, 1, 1]])
    else:
        raise ValueError("function must be 'sobel' or 'prewitt'")
   
    #Calculate the gradient and direction of the edge
    Gx = signal.convolve2d(filteredImg, Mx, mode='same', boundary='symm')
    Gy = signal.convolve2d(filteredImg, My, mode='same', boundary='symm')
    grad = np.sqrt(Gx**2 + Gy**2)
    
    #Change the direction to the closest one (0°, 45°, 90° or 135°)
    angle = np.rad2deg(np.arctan2(Gy, Gx)) % 360
    direction = np.floor(angle / 45) * 45 #Better results than np.round for some reason
    direction = direction % 180 
    
    #Gradient non-maximum suppression (edge thinning)
    nms = grad.copy()
    for i, j in np.argwhere(grad):
        if i == 0 or i == grad.shape[0]-1 or j == 0 or j == grad.shape[1]-1:
            continue
        match direction[i, j]:
            case 0:
                if grad[i, j] < grad[i, j-1] or grad[i, j] < grad[i, j+1]:
                    nms[i,j]=0
            case 45:
                if grad[i, j] < grad[i+1, j-1] or grad[i, j] < grad[i-1, j+1]:
                    nms[i,j]=0
            case 90:
                if grad[i, j] < grad[i-1, j] or grad[i, j] < grad[i+1, j]:
                    nms[i,j]=0
            case 135:
                if grad[i, j] < grad[i+1, j+1] or grad[i, j] < grad[i-1, j-1]:
                    nms[i,j]=0
    grad = nms
    #Normalize the resulting gradient array (+1e-8 in order to avoid undef)
    grad = grad /(grad.max() + 1e-8) * 255
    high = np.percentile(grad, nth)
    low = 0.9*high
    strong = 255
    weak = 75
    edges = np.zeros_like(grad, dtype=np.uint8)
    edges[np.where(grad >= low)] = weak
    edges[np.where(grad >= high)] = strong

    # 7) Hysteresis
    edges = hysteresis(edges, weak, strong)
    return edges
