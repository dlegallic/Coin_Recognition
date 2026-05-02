import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy import ndimage
import matplotlib.patches as patches

import edgeDetection
import houghTransform
import optimization
import utils 

import json
from pathlib import Path
import os
import shutil

def histogrammeLabels(path):
    p = Path(path)  
    valeurs = {}
    for label in p.iterdir(): 
        with open(str(label), "r") as jsonfile: 
            data = json.load(jsonfile)
            for shape in data['shapes']:
                if shape['label'] in valeurs:
                    valeurs[shape['label']]+=1
                else :
                    valeurs[shape['label']]=1
    
    plt.bar(range(len(valeurs)), list(valeurs.values()), align='center')
    plt.xticks(range(len(valeurs)), list(valeurs.keys()))

#histogramme('../bases/base_test/labels_test')
#histogramme('../bases/base_validation/labels_validation')


#TOUTES LES FONCTIONS AUXILIAIRES CI-DESSOUS SONT ASSEZ MAUVAISES IL FAUT LES REECRIRE

def showCircles(circles, radii, image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    for circle in circles:
        x=circle[1]
        y=circle[0]
        rad=radii[int(circle[2])]
        #print(circle[3])
        circle = Circle((x, y), radius=rad, edgecolor='red', facecolor='none', linewidth=1)
        ax.add_patch(circle)
    plt.show()
    
def showRectangle(circles, radii, image, factor):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    for circle in circles:
        y,x,r = circle
        rad=radii[int(r)]  * factor 
        rec = patches.Rectangle((x-rad,y-rad), rad*2, rad*2, edgecolor='red', facecolor='none', linewidth=1)
        ax.add_patch(rec)
    plt.show()
    
def circleToRectangle(circle, radii, factor):
    y,x,r = circle
    rad = radii[int(r)] * factor
    startX = x-rad
    startY = y-rad
    startPoint=(startX, startY)
    
    endX = x+rad
    endY = y+rad
    endPoint = (endX, endY)
    return(startPoint,endPoint)

def processImg(image):
    imgSize = 50000 #We will rarely exceed coins of more than 60px radius
    radii = np.arange(8,60)
    
    #pre-process
    array = utils.imgToArray(image)
    downscaledImg, factor = utils.downscale(array,imgSize)
    downscaledImg = utils.normalizeImg(downscaledImg)
    
    #Isolating the edges with canny filter
    edgeImg = edgeDetection.cannyFilter(downscaledImg)
    smoothEdge = edgeDetection.gaussianBlur(edgeImg)
    
    #Computing the associated houghSpace
    houghSpace = houghTransform.fastCHT(smoothEdge, radii)
    normalizedHough = utils.normalizeAcc(houghSpace, radii)
    
    #Identifying the circles
    circles = utils.nms3d(normalizedHough, threshold=np.percentile(normalizedHough, 99.99))
    showCircles(circles, radii, downscaledImg)
    
    #circle as [i,j,r] with i,j position is the original image
    circleResults = [[circle * factor for circle in sub[:-2]] + [sub[-2]] for sub in circles]
    showRectangle(circleResults, radii, array, factor)
    rectangleResults = [circleToRectangle(circle, radii, factor) for circle in circleResults]
    print(rectangleResults)
    return rectangleResults

processImg('../testImage/1.jpeg')


    