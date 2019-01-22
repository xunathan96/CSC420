import numpy as np
from skimage import io
from scipy import ndimage

from gaussian import *

def q2B():
    G = gaussian_kernel(2)
    print(G)

def q2C():
    image = io.imread('images/waldo.png')[:,:,:3]
    G = gaussian_kernel(2)

    # Convolve each RGB dimension
    image[:,:,0] = ndimage.convolve(image[:,:,0], G, mode='constant', cval=0.0)
    image[:,:,1] = ndimage.convolve(image[:,:,1], G, mode='constant', cval=0.0)
    image[:,:,2] = ndimage.convolve(image[:,:,2], G, mode='constant', cval=0.0)

    io.imshow(image)
    io.show()


if __name__ == '__main__':
    q2B()
    q2C()
