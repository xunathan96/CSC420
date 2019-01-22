import numpy as np
from correlation import *

def convolution(image, filter):

    # flip filter on both its axis
    filter = np.flip(filter, 0)
    filter = np.flip(filter, 1)

    # filter applied in convolution is just the filter
    # flipped on both axis and applied in correlation
    return cross_correlation(image, filter)


# CONVOLUTION OF A RGB IMAGE AND A 2D FILTER
# image: 3 dimensions
# filter: 1 dimension
def convolution_RGB_image(image, filter):

    if image.ndim == 2:
        # GRAYSCALE
        filtered_image = convolution(image, filter)
    elif image.ndim == 3:
        # RGB
        filtered_image = np.empty_like(image)
        filtered_image[:,:,0] = convolution(image[:,:,0], filter)
        filtered_image[:,:,1] = convolution(image[:,:,1], filter)
        filtered_image[:,:,2] = convolution(image[:,:,2], filter)

    return filtered_image

# GENERAL 3D CONVOLUTION
# image: 3+ dimensions
# filter: 3 dimensions
def convolution_3D(image, filter):

    # flip filter on all its axis
    filter = np.flip(filter, 0)
    filter = np.flip(filter, 1)
    filter = np.flip(filter, 2)

    # filter applied in convolution is just the filter
    # flipped on both axis and applied in correlation
    return cross_correlation_3D(image, filter)
