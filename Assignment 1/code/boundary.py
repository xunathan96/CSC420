import numpy as np

def crop_filter(filter):
    height, width = filter.shape
    # Crop filter to be odd x odd
    if height%2 == 0:
        height = height - 1
    if width%2 == 0:
        width = width - 1
    filter = filter[:height, :width]
    return filter

def zero_pad(image, filter):
    height, width = filter.shape
    j = int((height-1)/2)
    k = int((width-1)/2)

    pad_axis_1 = (j,j)      # Pad j pixels before and after axis 1
    pad_axis_2 = (k,k)
    padding = (pad_axis_1, pad_axis_2)

    frame = np.pad(image, padding, mode='constant', constant_values=0)
    return frame, j, k

def zero_pad_3D(image, filter):
    height, width, depth = filter.shape
    j = int((height-1)/2)
    k = int((width-1)/2)
    l = int((depth-1)/2)

    pad_axis_1 = (j,j)      # Pad j pixels before and after axis 1
    pad_axis_2 = (k,k)
    pad_axis_3 = (l,l)
    padding = (pad_axis_1, pad_axis_2, pad_axis_3)

    frame = np.pad(image, padding, mode='constant', constant_values=0)
    return frame, j, k, l

def zero_pad_extend(image, filter):
    height, width = filter.shape

    pad_axis_1 = (height,height)      # Pad j pixels before and after axis 1
    pad_axis_2 = (width,width)
    padding = (pad_axis_1, pad_axis_2)

    frame = np.pad(image, padding, mode='constant', constant_values=0)
    return frame, height, width
