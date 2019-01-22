import numpy as np
from scipy import ndimage

def gradient(image):

    sobel_x = np.array([
        [-1, 0, +1],
        [-2, 0, +2],
        [-1, 0, +1]
    ])
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [+1, +2, +1]
    ])

    G_x = ndimage.correlate(image, sobel_x, mode='nearest')
    G_y = ndimage.correlate(image, sobel_y, mode='nearest')
    G = np.sqrt(G_x**2 + G_y**2)

    tan_theta = G_y/G_x             #ignore /0 warnings
    theta = np.arctan(tan_theta)

    # NORMAIZE THE MATRIX TO GRAYSCALE (0-1)
    G_max = G.max()
    G = G/G_max

    return G, G_x, G_y, theta
