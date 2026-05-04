import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import matplotlib.patches as patches

import edgeDetection
import houghTransform
import optimization
import utils 
import opencvComparison as cv

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


#Not the most optimized functions, but it works
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
def processImg(image, graphicEnabled=False):
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
    if graphicEnabled:
        showCircles(circles, radii, downscaledImg)
        #showRectangle(circleResults, radii, array, factor)
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

#Compare one image to one json
def compareImgToJson(imagePath, jsonPath, process, p1, p2, graphicEnabled):
    if(process=='manual'):
        recList1 = processImg(imagePath, graphicEnabled)
    elif(process=='opencv'):
        recList1 = cv.processImg(imagePath, p1, p2, graphicEnabled)
    else:
        raise ValueError("process must be 'manual' or 'opencv'")
        
    points = []
    with open(jsonPath, "r") as jsonfile: 
        data = json.load(jsonfile)
        H = data['imageHeight']
        W = data['imageWidth']
        for shape in data['shapes']:
            points.append(shape['points'])
    recList2 = np.array(points, dtype=np.uint32)
    return IoU(recList1,recList2,H,W)

#Compare the sets
def iterate(imageFolder, jsonFolder, process='manual', p1=150, p2=30, graphicEnabled=False):
    images_dir = Path(imageFolder)
    json_dir = Path(jsonFolder)
    
    elt_nb = 1
    mean = 0
    precision = 0 
    for image_path in images_dir.iterdir():
        print("Image n°",elt_nb)
        json_path = json_dir / (image_path.stem + ".json")
        IoU = compareImgToJson(image_path, json_path, process, p1, p2, graphicEnabled)
        print("Score IoU : ",IoU)
        if(IoU>0.5):precision+=1
        mean += IoU
        elt_nb += 1
    mean /= elt_nb
    precision /= elt_nb
    print("Score IoU moyen : ",mean)
    print("Precision : ", precision)
    return (mean, precision)
    
#NOT WORKING / POORLY
#Was meant to optimize the opencv CHT, it is ultimately left to the user
def gradientAscentOPENCV(steps, startp1=110, startp2=70):
    p1, p2 = startp1, startp2
    pas = 5
    lr = 5000
    path = '../bases/base_test/images_test'
    labels = '../bases/base_test/labels_test'
    for _ in range(steps):
        IoU, _ = iterate(path, labels, p1, p2, 'opencv')
        print(IoU)

        IoUp1, _ = iterate(path, labels, p1 + pas, p2, 'opencv')
        print(IoUp1)
        IoUm1, _ = iterate(path, labels, p1 - pas, p2, 'opencv')
        print(IoUm1)
        IoUp2, _ = iterate(path, labels, p1, p2 + pas, 'opencv')
        print(IoUp2)
        IoUm2, _ = iterate(path, labels, p1, p2 - pas, 'opencv')
        print(IoUm2)
        
        dIoU_dp1 = (IoUp1 - IoUm1) / (2 * pas)
        dIoU_dp2 = (IoUp2 - IoUm2) / (2 * pas)

        p1 += lr * dIoU_dp1
        p2 += lr * dIoU_dp2

        p1 = int(round(p1))
        p2 = int(round(p2))
        print(p1,p2)
    return (p1,p2)

    