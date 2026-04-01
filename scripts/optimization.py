import numpy as np
import random
import matplotlib.pyplot as plt

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


