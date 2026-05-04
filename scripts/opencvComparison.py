import cv2
import numpy as np
import matplotlib.pyplot as plt

#Mostly the same as the manual one, only with opencv's own methods.
def processImg(imagePath, p1, p2, graphicEnabled=False):
    img = cv2.imread(str(imagePath), cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read"
    
    #Downscales
    h, w = img.shape[:2]
    scale = min(400 / max(h, w), 1.0)
    img = cv2.resize(img, None, fx=scale, fy=scale)

    img = cv2.GaussianBlur(img, (9, 9), 2)
    edge = cv2.Canny(img, 50, p1)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    #Here p1 and p2 have the most influence on the result
    circles = cv2.HoughCircles(
        edge,
        cv2.HOUGH_GRADIENT,
        dp=1.3,
        minDist=30,
        param1=p1,   
        param2=p2,   
        minRadius=20,
        maxRadius=120
    )

    if circles is None:
        #print("No circles detected")
        return []
    circles = np.uint16(np.around(circles))

    #Converts circle to rectangles
    rec = []
    for x, y, r in circles[0]:
        cv2.circle(cimg, (x, y), r, (0, 255, 0), 2)
        cv2.circle(cimg, (x, y), 2, (0, 0, 255), 3)
        
        startX = int((x-r)/scale)
        startY = int((y-r)/scale)
        startPoint=(startX, startY)
        endX = int((x+r)/scale)
        endY = int((y+r)/scale)
        endPoint = (endX, endY)
        rec.append([startPoint, endPoint])
    rec = np.array(rec)
    
    #Shows the resulting rectangles if needed
    if graphicEnabled:
        plt.figure(figsize=(8, 6))
        plt.imshow(cimg)
        plt.title("detected circles")
        plt.axis("off")
        plt.show()
    return rec