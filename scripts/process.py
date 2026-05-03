import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import matplotlib.patches as patches

import edgeDetection
import houghTransform
import optimization
import utils 

import json
from pathlib import Path

#Just to see the data, histogram of coin types
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

#Shows the cirlces superposed on the image
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

#Shows the rectangle derived from the circle, superposed on the image
def showRectangle(circles, radii, image, factor):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    for circle in circles:
        y,x,r = circle
        rad=radii[int(r)]  * factor 
        rec = patches.Rectangle((x-rad,y-rad), rad*2, rad*2, edgecolor='red', facecolor='none', linewidth=1)
        ax.add_patch(rec)
    plt.show()

#Converts a [i,j,r] circle to a [[x1,y1],[x2,y2]] rectangle
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

#Finds rectangles around coins in a given image 
def processImg(image):
    #Parameters
    #==========================================#
    imgSize = 50000 #Size = Height*Width
    radii = np.arange(3,70) #List of test radii
    threshold=99.84 #Threshold applied in the 3D NMS
    #==========================================#
    
    #Pre-process
    array = utils.imgToArray(image)
    downscaledImg, factor = utils.downscale(array,imgSize)
    downscaledImg = np.where(downscaledImg > np.percentile(downscaledImg, 92), 0.98*downscaledImg, 1.02*downscaledImg)
    downscaledImg = utils.normalizeImg(downscaledImg)
    
    #Isolating the edges with canny filter
    edgeImg = edgeDetection.cannyFilter(downscaledImg)
    smoothEdge = edgeDetection.gaussianBlur(edgeImg)
    
    #Computing the associated houghSpace
    houghSpace = houghTransform.fastCHT(smoothEdge, radii)
    normalizedHough = optimization.normalizeAcc(houghSpace, radii)
    
    #Identifying the circles
    circles = optimization.nms3d(normalizedHough, threshold=np.percentile(normalizedHough, threshold))
    
    #Circles as [i,j,r] with i,j position is the original image
    circleResults = [[circle * factor for circle in sub[:-2]] + [sub[-2]] for sub in circles]
    rectangleResults = np.array([circleToRectangle(circle, radii, factor) for circle in circleResults], dtype=(np.uint32))
    
    #Show the identified circles and rectangles
    #showCircles(circles, radii, downscaledImg)
    showRectangle(circleResults, radii, array, factor)
    return rectangleResults

#Makes a mask, 1 inside rectangles 0 outside
def rasterize(rects, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    for r in rects:
        x1, y1 = r[0]
        x2, y2 = r[1]
        
        #labeled points are not always top-left to bottom-right
        startx, endx = sorted((x1, x2))
        starty, endy = sorted((y1, y2))
        
        mask[int(starty):int(endy), int(startx):int(endx)] = 1
    return mask

#Compute IoU using logical and between masks
def IoU(rects1, rects2, H, W):
    m1 = rasterize(rects1, H, W)
    m2 = rasterize(rects2, H, W)

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return intersection / (union + 1e-9)


def compareImgToJson(imagePath, jsonPath):
    recList1 = processImg(imagePath)
    points = []
    with open(jsonPath, "r") as jsonfile: 
        data = json.load(jsonfile)
        H = data['imageHeight']
        W = data['imageWidth']
        for shape in data['shapes']:
            points.append(shape['points'])
    recList2 = np.array(points, dtype=np.uint32)
    return IoU(recList1,recList2,H,W)

def iterate(imageFolder, jsonFolder):
    images_dir = Path(imageFolder)
    json_dir = Path(jsonFolder)
    
    elt_nb = 1
    mean = 0
    precision = 0 
    for image_path in images_dir.iterdir():
        print("Image n°",elt_nb)
        json_path = json_dir / (image_path.stem + ".json")
        IoU = compareImgToJson(image_path, json_path)
        print("Score IoU : ",IoU)
        if(IoU>0.5):precision+=1
        mean += IoU
        elt_nb += 1
    mean /= elt_nb
    precision /= elt_nb
    return (mean, precision)
    
#print(compareImgToJson('../bases/base_test/images_test/1000016159.jpg', '../bases/base_test/labels_test/1000016159.json'))
print(iterate('../bases/base_test/images_test', '../bases/base_test/labels_test'))



    