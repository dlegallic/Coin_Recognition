import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter

#Fonction differentiable en tout point, représentée par un array
def gradientDescent1D(functionArray, learningRate, steps):
    a  = len(functionArray)/2
    for n in range(steps):
        index = (int)(a)
        if index <= 0 or index >= len(functionArray) - 1:
            break
        
        grad = (functionArray[index+1]-functionArray[index-1])/2
        a =  a - learningRate*grad[index]
        
        a = max(1, min(len(functionArray) - 2, a))
    return(int)(a)


def hessian(imageArray):
    Mx = [[-1, 0, 1],
          [-1, 0, 1],
          [-1, 0, 1]]
    
    My = [[-1,-1,-1],
          [ 0, 0, 0],
          [ 1, 1, 1]]

    height, width = imageArray.shape
    
    Gxx = signal.convolve2d(signal.convolve2d(imageArray, Mx, mode='valid'), Mx, mode='valid')
    Gxy = signal.convolve2d(signal.convolve2d(imageArray, Mx, mode='valid'), My, mode='valid')
    Gyy = signal.convolve2d(signal.convolve2d(imageArray, My, mode='valid'), My, mode='valid')
    
    detH = Gxx * Gyy - Gxy**2
    return detH


def normalizeAcc(acc, radii):
    #Normalize the accumulator along the r axis to avoid too big a bias toward higher radius
    norm = 1.0/(2 * np.pi * np.power(radii,0.5) + 1e-8)
    acc = acc * norm[None, None, :]
    return acc


#Greedy NMS in 3D accumulator. 
#Returns a list of circles [x,y,r_idx,score], be careful to convert r_idx to r
def nms3d(houghSpace, xy_radius=18, r_radius=30, sigma=2.4, threshold=1):
    acc = houghSpace.copy()
    acc = gaussian_filter(acc, sigma=(sigma, sigma, 1))
    H, W, R = acc.shape
    circles = []
    while True:
        idx = np.unravel_index(np.argmax(acc), acc.shape)
        x, y, r_idx = idx
        score = acc[x, y, r_idx]

        #threshold to select circles
        if score <= threshold:
            break

        circles.append((x, y, r_idx, score))

        # Select the intervals in each dimension that will be zeroed out
        x0 = max(0, x - xy_radius)
        x1 = min(H, x + xy_radius + 1)

        y0 = max(0, y - xy_radius)
        y1 = min(W, y + xy_radius + 1)

        r0 = max(0, r_idx - r_radius)
        r1 = min(R, r_idx + r_radius + 1)

        # zero out neighborhood
        acc[x0:x1, y0:y1, r0:r1] *= 0.1
    return np.array(circles)
