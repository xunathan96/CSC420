import numpy as np
from boundary import *

# 2D CORRELATION ----------------------------------------------------
def correlation(image, filter, i, j):
    height, width = filter.shape
    k = int((height-1)/2)
    l = int((width-1)/2)

    # vectorize area around point i,j
    mask = image[i-k:i+k+1, j-l:j+l+1].flatten()
    filter = filter.flatten()

    res = np.dot(mask, filter)
    return res

def cross_correlation(image, filter):
    height, width = image.shape
    frame, col_pad, row_pad = zero_pad(image, filter)

    res = np.empty_like(frame)
    # Traverse all pixels of the image and calculate the correlation at each point
    for i in range(col_pad, col_pad+height):
        for j in range(row_pad, row_pad+width):
            res[i, j] = correlation(frame, filter, i, j)

    # Return the original size of the image
    return res[col_pad:col_pad+height, row_pad:row_pad+width]

# 3D CORRELATION ----------------------------------------------------
def correlation_3D(image, filter, i, j, k):
    height, width, depth = filter.shape
    p = int((height-1)/2)
    q = int((width-1)/2)
    r = int((depth-1)/2)

    # vectorize 3D shape around point i,j,k
    mask = image[i-p:i+p+1, j-q:j+q+1, k-r:k+r+1].flatten()
    filter = filter.flatten()

    res = np.dot(mask, filter)
    return res

def cross_correlation_3D(image, filter):
    height, width, depth = image.shape
    frame, col_pad, row_pad, depth_pad = zero_pad_3D(image, filter)

    res = np.empty_like(frame)
    # Traverse all elements of the 3D matrix and calculate the 3D correlation at each point
    for i in range(col_pad, col_pad+height):
        for j in range(row_pad, row_pad+width):
            for k in range(depth_pad, depth_pad+depth):
                res[i, j, k] = correlation_3D(frame, filter, i, j, k)

    # Return the original size of the image
    return res[col_pad:col_pad+height, row_pad:row_pad+width, depth_pad:depth_pad+depth]
