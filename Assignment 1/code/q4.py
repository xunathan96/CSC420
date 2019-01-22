import numpy as np
from skimage import io
from matplotlib import pyplot as plt

from canny_edge_detector import *


def q4():
    image = io.imread('images/waldo.png', as_gray=True)

    CED = canny_edge_detector(image, std=1)

    # PLOT GRAYSCALE and CANNY EDGE DETECTED IMAGES
    fig, axis = plt.subplots(1, 2)      # nrows, ncols
    plt.subplot(1,2,1)                  # nrows, ncols, index
    plt.imshow(image, cmap='gray')

    plt.subplot(1,2,2)
    plt.imshow(CED, cmap='gray')

    plt.show()







if __name__ == '__main__':
    q4()
