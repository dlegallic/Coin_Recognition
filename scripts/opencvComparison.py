import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import json
from pathlib import Path
 


def findCircles(imagePath):
    img = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
    imgSize=len(img)
    ratio = len(img)/len(img[0])
    img = cv.resize(img, (300, (int)(ratio*300)))
    
    assert img is not None, "file could not be read, check with os.path.exists()"
    img = cv.GaussianBlur(img,(9,9),0)
     
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=0,maxRadius=0)
     
    for circle in circles[0]:
        circle[0] *= imgSize/300
        circle[1] *= (imgSize*ratio)/300
        circle[2] *= imgSize/300
    return circles


pImg = Path('../bases/base_test/images_test')
#pLbl = Path('../bases/base_test/labels_test')
foundPoints=[]
refPoints = []
for image in pImg.iterdir(): 
    foundPoints.append(findCircles(str(image)))
print(foundPoints)
        