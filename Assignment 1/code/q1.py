import numpy as np
from skimage import io
from convolution import *
from scipy import ndimage

def q1A():
    image = io.imread('images/waldo.png', as_gray=True)
    filter = np.array([
        [1,4,7,4,1],
        [4,16,26,16,4],
        [7,26,41,26,7],
        [4,16,26,16,4],
        [1,4,7,4,1],
    ])*(1/273)

    filtered_image = convolution(image, filter)
    #filtered_image = ndimage.convolve(image, filter, mode='constant', cval=0.0)

    io.imshow(filtered_image)
    io.show()

def q1B():
    image = io.imread('images/waldo.png')[:,:,:3]
    filter = np.array([
        [
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ],
        [
            [0,0,0],
            [0,1,0],
            [0,0,0]
        ],
        [
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ]
    ])

    # 3D Convolution
    # image: 3+ dimensions
    # filter: 3 dimensions
    res = convolution_3D(image, filter)

    # if we want to convolve a 2D filter with an RGB image
    # image: 3 dimensions (RGB)
    # filter: 2 dimensions
    # res = convolution_RGB_image(image, filter)

    io.imshow(res)
    io.show()

if __name__ == '__main__':
    q1A()
    q1B()
