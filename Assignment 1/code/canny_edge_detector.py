import numpy as np
from skimage.filters import *
from gradient import *

def round_direction(theta):
    # Convert to degrees b/c easier to understand
    angle = (theta/np.pi)*180

    # Round each angle to 45 degrees
    angle = np.round(angle/45.0) * 45
    return angle

def canny_edge_detector(image, std):
    # APPLY GAUSSIAN SMOOTHING
    image = gaussian(image, sigma=std)

    # FIND IMAGE GRADIENT
    G, G_x, G_y, theta = gradient(image)
    angle = round_direction(theta)

    # Pad one extra pixel to avoid out of index error
    padding = (1,)
    G = np.pad(G, padding, mode='constant', constant_values=0)
    height, width = G.shape

    # NON MAXIMUM SUPRESSION
    for i in range(1, height-1):
        for j in range(1, width-1):
            dir = angle[i-1, j-1]
            if dir == 0:
                if not G[i, j] > max(G[i, j+1], G[i, j-1]):
                    G[i, j] = 0
            elif dir == 45:
                if not G[i, j] > max(G[i+1, j+1], G[i-1, j-1]):
                    G[i, j] = 0
            elif dir == -45:
                if not G[i, j] > max(G[i-1, j+1], G[i+1, j-1]):
                    G[i, j] = 0
            elif dir == 90 or dir == -90:
                if not G[i, j] > max(G[i+1, j], G[i-1, j]):
                    G[i, j] = 0

    # HYSTERISIS THRESHOlDING
    # applied to only minVal threshold
    minVal = 0.5
    G[G < minVal] = 0

    return G[1:-1, 1:-1]    # Return original sized image
