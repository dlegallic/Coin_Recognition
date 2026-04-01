import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def downscale(image, size):
    #Low pass filter (smoothing)
    img = plt.imread(image)
    grayImg = np.round(0.299 * img[:, :, 0] +
                        0.587 * img[:, :, 1] +
                        0.114 * img[:, :, 2]).astype(np.uint8)
    K=np.array([[0.33,  0.33,  0.33],
               [0.33,  0.33,  0.33],
               [0.33,  0.33,  0.33]])
    filteredImg = signal.convolve2d(grayImg, K)
    
    while(filteredImg.size>size):
          filteredImg = filteredImg[0::2,0::2]
    return filteredImg


img = plt.imread('../testImage/euros.jpeg')
plt.imshow(img,  cmap='gray')
plt.imshow(downscale('../testImage/euros.jpeg', 50000), cmap='gray')
plt.show()