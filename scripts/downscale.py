import numpy as np
from scipy import signal

def downscale(imageArray, size):
    #Low pass filter (smoothing)
    K=np.array([[0.33,  0.33,  0.33],
               [0.33,  0.33,  0.33],
               [0.33,  0.33,  0.33]])
    filteredImg = signal.convolve2d(imageArray, K, mode='valid')
    
    while(filteredImg.size>size):
          filteredImg = filteredImg[0::2,0::2]
    return filteredImg