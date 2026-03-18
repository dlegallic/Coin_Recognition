import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


#a vectoriser parce que niveau vitesse c'est catastrophique
def hough_transformSLOW(image):
    img = plt.imread(image)
    if len(img.shape) == 3:
        img = img[:, :, 0]
    h,w = img.shape
    
    rho_max = (int)(np.sqrt((h/2)**2 + (w/2)**2))
    H=np.zeros((180,2*rho_max))
    
    for i in range(h):
        for j in range(w):
            x = j-(int)(w/2)
            y = h-i-(int)(h/2)
            if (img[i][j]>0):
                for theta in range(180):
                    theta_rad = np.deg2rad(theta)
                    rho = (int)(x*np.cos(theta_rad) + y*np.sin(theta_rad))
                    rho_index = rho + rho_max
                    H[theta, rho_index] += 1
    return H


#trouver le ou les max, -> transformer en coordonnées pour tracer une ligne.
#Il faut implémenter une recherche de max efficace
def hough_line(hough_transform):
    inputHough = plt.imread(hough_transform)
    if len(inputHough.shape) == 3:
        inputHough = inputHough[:, :, 0]
    h,w = inputHough.shape
    max = 0
    index = (0,0)
    for i in range(h):
        for j in range(w):
            if (inputHough[i][j]>max):
                max = inputHough[i][j]
                index = (i,j)
    p1 = (0,(int)(index[1]/np.sin(np.deg2rad(index[0]))))
    p2 = ((int)(index[1]/np.cos(np.deg2rad(index[0]))), 0)
    return (p1,p2)


plt.imsave("houghline.png", hough_transformSLOW("line.png"), cmap='gray')
print(hough_line("houghline.png"))
plt.show()