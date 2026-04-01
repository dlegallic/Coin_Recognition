import numpy as np
import matplotlib.pyplot as plt
from scipy import signal



def slowLineHoughTransform(imageArray):
    h,w = imageArray.shape
    rho_max = (int)(np.sqrt((h/2)**2 + (w/2)**2))
    H=np.zeros((2*rho_max,180))
    
    for i in range(h):
        for j in range(w):
            x = j-(int)(w/2)
            y = h-i-(int)(h/2)
            if (imageArray[i][j]>0):
                for theta in range(180):
                    theta_rad = np.deg2rad(theta)
                    rho = (int)(x*np.cos(theta_rad) + y*np.sin(theta_rad))
                    rho_index = rho + rho_max
                    H[rho_index][theta] += 1
    return H

def vectorizedLineHoughTransform(imageArray):
    h, w = imageArray.shape
    ys, xs = np.nonzero(imageArray > 0)
    x = xs - w // 2
    y = h - ys - h // 2
    
    thetas = np.deg2rad(np.arange(180))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    rho_max = int(np.sqrt((h/2)**2 + (w/2)**2))
    rho = x[:, None] * cos_t[None, :] + y[:, None] * sin_t[None, :]
    rho = rho.astype(int)
    rho_idx = rho + rho_max

    H = np.zeros((2 * rho_max,180))
    theta_idx = np.tile(np.arange(180), (rho.shape[0], 1))
    np.add.at(H, (rho_idx, theta_idx), 1)
    return H

def bresenhamCircle(array, p, radius):
    h = len(array)
    w = len(array[0])

    def plot(x, y):
        if 0 <= x < w and 0 <= y < h:
            array[y][x] += 1

    x = 0
    y = radius
    m = 5 - 4 * radius

    while x <= y:
        plot(p[0] + x, p[1] + y)
        plot(p[0] + y, p[1] + x)
        plot(p[0] - x, p[1] + y)
        plot(p[0] - y, p[1] + x)
        plot(p[0] + x, p[1] - y)
        plot(p[0] + y, p[1] - x)
        plot(p[0] - x, p[1] - y)
        plot(p[0] - y, p[1] - x)

        if m > 0:
            y -= 1
            m -= 8 * y
        x += 1
        m += 8 * x + 4
    return array
    
    
def CHT(imageArray):
    h, w = imageArray.shape
    ys, xs = np.nonzero(imageArray > 0)
    points = list(zip(xs, ys))
    rho_max = int(np.sqrt((h/2)**2 + (w/2)**2))

    H = np.zeros((rho_max, h, w), dtype=np.uint8)
    for rho in range (rho_max):
        for point in points:
            bresenhamCircle(H[rho], point, rho)
    return H

def lessAccurateCHT(image, pas):
    img = plt.imread(image)
    if len(img.shape) == 3:
        img = img[:, :, 0]

    h, w = img.shape
    ys, xs = np.nonzero(img > 0)
    points = list(zip(xs, ys))
    rho_max = int(np.sqrt((h/2)**2 + (w/2)**2))

    H = np.zeros((rho_max//pas, h, w), dtype=np.uint8)
    for rho in range (0, rho_max, pas):
        for point in points:
            bresenhamCircle(H[rho], point, rho)
    return H

def fixedCHT(imageArray, rho):
    h, w = imageArray.shape
    ys, xs = np.nonzero(imageArray > 0)
    points = list(zip(xs, ys))
    
    H = np.zeros((h, w), dtype=np.uint8)
    for point in points:
        bresenhamCircle(H, point, rho)
    return H

#really bad
def findLocalExtrema1D(array, nbTry, maxTime):
    startPoints = np.linspace(1, len(array)-2, nbTry, dtype=int)
    endPoints = set()
    for point in startPoints:
        time = 0
        while(time<maxTime):
            if (array[point-1]>array[point] and point>1):
                point-=1
            elif (array[point+1]>array[point] and point<len(array)-2):
                point+=1
            time +=1
        endPoints.add(point)
    return endPoints

def findLocalExtrema(array):
    return signal.find_peaks(array);

def findLines(inputHough):
    indexList = findLocalExtrema(inputHough)
    lines = []
    for index in indexList:
        p1 = (0,(int)(index[1]/np.sin(np.deg2rad(index[0]))))
        p2 = ((int)(index[1]/np.cos(np.deg2rad(index[0]))), 0)
        lines.append((p1,p2))
    return lines

def findCircle(inputHough, rho):
    indexList = findLocalExtrema(inputHough)
    circles = []
    for index in indexList:
        circles.append((index,rho))
    return circles
