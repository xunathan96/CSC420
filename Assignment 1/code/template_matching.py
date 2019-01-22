import numpy as np
from boundary import *
from skimage import io

def normalized_correlation(image, filter, i, j):
    height, width = filter.shape

    mask = image[i:i+height, j:j+width].flatten()
    filter = filter.flatten()

    m_dot_f = np.dot(mask, filter)
    norm_m = np.linalg.norm(mask)
    norm_f = np.linalg.norm(filter)

    if (norm_m * norm_f)==0:
        res = 0
    else:
        res = m_dot_f/(norm_m * norm_f)
    return res

def normalized_cross_correlation(image, filter):
    height, width = image.shape
    frame, col_pad, row_pad = zero_pad_extend(image, filter)

    res = np.empty_like(frame)
    for i in range(col_pad, col_pad+height):
        for j in range(row_pad, row_pad+width):
            res[i, j] = normalized_correlation(frame, filter, i, j)

    # Return the original size of the image
    return res[col_pad:col_pad+height, row_pad:row_pad+width]

def match_template(image, filter):

    similarity = normalized_cross_correlation(image, filter)

    max_index = similarity.argmax()
    i, j = np.unravel_index(max_index, similarity.shape)

    height, width = filter.shape
    corners = np.array([
        [i, j],
        [i+height-1, j],
        [i+height-1, j+width-1],
        [i, j+width-1],
        [i, j]
    ])

    return similarity, corners
