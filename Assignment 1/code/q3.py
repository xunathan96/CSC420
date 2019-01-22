import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from gradient import *
from template_matching import *

def q3A():
    image = io.imread('images/waldo.png', as_gray=True)
    filter = io.imread('images/template.png', as_gray=True)

    G_image, G_x_image, G_y_image, theta_image = gradient(image)
    G_filter, G_x_filter, G_y_filter, theta_filter = gradient(filter)

    # PLOT BOTH IMAGE GRADIENTS
    fig, axis = plt.subplots(1, 2)      # nrows, ncols
    plt.subplot(1,2,1)                  # nrows, ncols, index
    plt.imshow(G_image, cmap='gray')

    plt.subplot(1,2,2)
    plt.imshow(G_filter, cmap='gray')

    plt.show()


def q3B():
    image = io.imread('images/waldo.png', as_gray=True)
    image_color = io.imread('images/waldo.png')
    filter = io.imread('images/template.png', as_gray=True)

    # GET IMAGE GRADIENTS
    G_image, G_x_image, G_y_image, theta_image = gradient(image)
    G_filter, G_x_filter, G_y_filter, theta_filter = gradient(filter)

    # LOCALIZE IMAGE
    similarity, corners = match_template(G_image, G_filter)

    # PLOT SIMILARITY
    fig, axis = plt.subplots(1, 2)      # nrows, ncols
    plt.subplot(1,2,1)                  # nrows, ncols, index
    plt.imshow(similarity)

    # PLOT BOX AROUND TEMPLATE
    plt.subplot(1,2,2)
    plt.plot(corners[:, 1], corners[:, 0], 'b')
    plt.imshow(image_color)

    plt.show()


if __name__ == '__main__':
    q3A()
    q3B()
