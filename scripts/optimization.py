import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import signal

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
