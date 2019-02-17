import numpy as np
from skimage import io
from skimage.color import rgb2gray
from scipy import ndimage
from gradient import *

def minimum_seam(image):
    # FIND IMAGE GRADIENT
    G, G_x, G_y = gradient(image)
    height, width = G.shape

    # GENERATE DP LOOKUP TABLE
    E = np.full(G.shape, np.inf)        # Energy Table
    E[0] = G[0]
    P = np.empty_like(G, dtype=object)  # Seam Path Table

    # POPULATE LOOK UP TABLES
    for i in range(1, height):
        for j in range(width):
            E_left = E[i-1,j-1] if j-1 >=0 else np.inf
            E_right = E[i-1,j+1] if j+1 < width else np.inf
            E_center = E[i-1,j]

            # UPDATE ENERGY TABLE
            E[i,j] = G[i,j] + min(E_left, E_center, E_right)

            # UPDATE PATH TABLE
            if E_left == min(E_left, E_center, E_right):
                P[i,j] = (i-1, j-1)
            elif E_center == min(E_left, E_center, E_right):
                P[i,j] = (i-1, j)
            elif E_right == min(E_left, E_center, E_right):
                P[i,j] = (i-1, j+1)

    # FIND MINIMUM ENERGY PATH
    min_i = height-1
    min_j = np.argmin(E[height-1])
    seam_root = (min_i, min_j)

    # RETURN SEAM PATH AND SEAM ROOT INDEX
    return P, seam_root



def seam_carving(image):
    image_gray = rgb2gray(image)
    P, seam_index = minimum_seam(image_gray)

    carve_image = np.zeros_like(image[:,:-1,:])
    seam_image = np.copy(image)

    while True:
        # Get index of the pixel on the seam
        i, j = seam_index[0], seam_index[1]

        # Draw Seam Line in Red
        seam_image[i,j] = (255,0,0)

        # Remove pixel
        carve_image[i,:,0] = np.delete(image[i,:,0], j)
        carve_image[i,:,1] = np.delete(image[i,:,1], j)
        carve_image[i,:,2] = np.delete(image[i,:,2], j)

        if i != 0:  # Move to next pixel on seam path
            seam_index = P[i,j]
        else:       # Exit loop after entire seam is traversed
            break

    return seam_image, carve_image


def main():
    image = io.imread('images/water.jpg', as_gray=False)

    carve_image = np.copy(image)
    for i in range(80):
        seam_image, carve_image = seam_carving(carve_image)
        print(carve_image.shape)
        #io.imshow(seam_image)
        #io.show()

    io.imshow(image)
    io.show()

    io.imshow(seam_image)
    io.show()

    io.imshow(carve_image)
    io.show()


if __name__ == '__main__':
    print("Hello World")
    main()
