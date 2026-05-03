import numpy as np
from scipy import signal
from PIL import Image, ImageOps

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
    img = Image.open(image)
    img = ImageOps.exif_transpose(img)  #some images are rotated
    img = np.array(img)
    gray_img = np.round(0.299 * img[:, :, 0] +
                        0.587 * img[:, :, 1] +
                        0.114 * img[:, :, 2]).astype(np.uint8) 
    return gray_img

#Failed attempt
def imgToArrayByVar(image):
    #Load the image and convert it to a grayscale
    img = Image.open(image)
    img = ImageOps.exif_transpose(img)  #some images are rotated
    img = np.array(img)
    r = img[:,:,0]
    g = img[:,:,1]
    #There's no blue shade on coin (assumption)
    variances = [r.var(), g.var()]/(r.var()+g.var())
    gray_img = np.round(variances[0] * img[:, :, 0] +
                        variances[1] * img[:, :, 1] +
                        0 * img[:, :, 2]).astype(np.uint8) 
    return gray_img



def normalizeImg(imageArray):
    max_value = imageArray.max()
    normalizedImgArray = ((imageArray/max_value)*255)
    return normalizedImgArray.astype(int)

def histEQ(imageArray):
    hist, bins = np.histogram(imageArray.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    img_eq = cdf_normalized[imageArray].astype('uint16')
    return img_eq
