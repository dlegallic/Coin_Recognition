import matplotlib.pyplot as plt
import numpy as np

import edgeDetection
import downscale
import houghTransform
import utils 


def processImg(image):
    array = imgToArray(image)
    downscaledImg = downscale.downscale(array,50000)
    edgeImg = edgeDetection.cannyFilter(downscaledImg, "sobel")
    #houghSpace = houghTransform.fixedCHT(edgeImg, 66)
    #houghSpace = edgeDetection.gaussianBlur(houghSpace)
    #houghSpace = edgeDetection.prewittVectorized(houghSpace)
    #houghSpace = threshold(houghSpace, 25)
    return edgeImg

plt.imshow(processImg('../testImage/euros.jpeg'))
plt.show()
    