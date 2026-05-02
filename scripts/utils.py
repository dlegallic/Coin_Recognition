import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def downscale(imageArray, size):
    factor = 1
    
    #Low pass filter (smoothing)
    K=np.array([[0.33,  0.33,  0.33],
               [0.33,  0.33,  0.33],
               [0.33,  0.33,  0.33]])
    filteredImg = signal.convolve2d(imageArray, K, mode='same', boundary='symm')
    
    while(filteredImg.size>size):
          filteredImg = filteredImg[0::2,0::2]
          factor *= 2
    
    return (filteredImg, factor)

def threshold(imageArray, threshold):
    return np.where(imageArray>threshold, 255, 0)

def imgToArray(image):
    #Load the image and convert it to a grayscale
    img = plt.imread(image)
    gray_img = np.round(0.299 * img[:, :, 0] +
                        0.587 * img[:, :, 1] +
                        0.114 * img[:, :, 2]).astype(np.uint8) 
    return gray_img

def normalizeImg(imageArray):
    max_value = imageArray.max()
    normalizedImgArray = ((imageArray/max_value)*255)
    return normalizedImgArray.astype(int)


def normalizeAcc(acc, radii):
    #Normalize the accumulator along the r axis to avoid bias toward higher radius
    norm = 1.0/(2 * np.pi * radii +1e-8)
    acc = acc * norm[None, None, :]
    return acc


#Greedy NMS in 3D accumulator. 
#Returns a list of circles [x,y,r_idx,score], be careful to convert r_idx to r
def nms3d(houghSpace, xy_radius=10, r_radius=20, threshold=1):
    acc = houghSpace.copy()
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
        acc[x0:x1, y0:y1, r0:r1] = 0
    return np.array(circles)