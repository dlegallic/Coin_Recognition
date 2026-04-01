import matplotlib.pyplot as plt
import numpy as np

import edgeDetection
import downscale
import houghTransform

def threshold(imageArray, threshold):
    return np.where(imageArray>threshold, 255, 0)

def imgToArray(image):
    #Load the image and convert it to a grayscale
    img = plt.imread(image)
    gray_img = np.round(0.299 * img[:, :, 0] +
                        0.587 * img[:, :, 1] +
                        0.114 * img[:, :, 2]).astype(np.uint8) 
    return gray_img