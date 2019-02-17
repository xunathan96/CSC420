import numpy as np
from skimage import io
from skimage import draw
from skimage.color import rgb2gray
from scipy import ndimage
from gradient import *

# RETURNS CORNERNESS MAP OF ONLY LOCAL MAXIMUM
def find_local_maximum(R):
    height, width = R.shape
    Corners = np.zeros_like(R)

    for i in range(height):
        for j in range(width):
            thresh = max(
                R[i-1,j-1] if i-1>=0 and j-1>=0 else 0,
                R[i-1,j]   if i-1>=0 else 0,
                R[i-1,j+1] if i-1>=0 and j+1<width else 0,
                R[i,j-1]   if j-1>=0 else 0,
                R[i,j+1]   if j+1<width else 0,
                R[i+1,j-1] if i+1<height and j-1>=0 else 0,
                R[i+1,j]   if i+1<height else 0,
                R[i+1,j+1] if i+1<height and j+1<width else 0
            )
            if R[i,j] >= thresh:
                Corners[i,j] = R[i,j]
    return Corners

def harris_corner_detector(image, window_size, stddev, thresh):
    image_gray = rgb2gray(image)

    # CREATE GAUSSIAN WINDOW FILTER
    if window_size%2==0: window_size=window_size+1
    k = int((window_size-1)/2)
    window = np.zeros((window_size, window_size))
    window[k, k] = 1
    window = ndimage.gaussian_filter(window, sigma=stddev)

    # FIND IMAGE GRADIENT TERMS
    G, I_x, I_y = gradient(image_gray)
    I_xx = I_x * I_x
    I_yy = I_y * I_y
    I_xy = I_x * I_y

    # COMPUTE M
    M_11 = ndimage.correlate(I_xx, window, mode='nearest')
    M_12 = M_21 = ndimage.correlate(I_xy, window, mode='nearest')
    M_22 = ndimage.correlate(I_yy, window, mode='nearest')

    # COMPUTE CORNERNESS R
    alpha = 0.04
    R = (M_11 * M_22 - M_12 * M_21) - alpha * (M_11 + M_22)**2

    # THRESHOLD R
    threshold = thresh
    R[R<threshold] = 0

    # NON-LOCAL-MAXIMUM SUPRESSION
    Corners = find_local_maximum(R)

    return R, Corners



def main():
    image = io.imread('images/building.jpg', as_gray=False)
    height, width, depth = image.shape

    R, Corners = harris_corner_detector(image, 5, stddev=1, thresh=0.5)

    # DRAW CORNERS ONTO IMAGE
    for i in range(height):
        for j in range(width):
            if Corners[i,j] > 0:
                rr, cc = draw.circle(i,j, 2, image.shape)
                image[rr,cc,:] = (255,0,0)

    #io.imshow(R)
    #io.show()
    #io.imshow(Corners)
    #io.show()

    io.imshow(image)
    io.show()


if __name__ == '__main__':
    main()
