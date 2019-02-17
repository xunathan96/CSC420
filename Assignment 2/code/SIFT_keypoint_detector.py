import numpy as np
from skimage import io
from skimage import draw
from skimage.color import rgb2gray
from skimage.transform import pyramid_gaussian
from scipy import ndimage
from math import ceil

# n_scales: number of scales in the octave
# sigma: base standard devation for each octave
def create_scale_space(octave, sigma, n_scales):
    shape = (octave.shape[0], octave.shape[1], n_scales)
    I = np.zeros(shape)
    k = 2**(1./(n_scales-1))

    for i in range(n_scales):
        stddev = (k**i) * sigma
        I[:,:,i] = ndimage.gaussian_filter(octave, sigma=stddev)

    return I

def difference_of_gaussians(I):
    shape = (I.shape[0], I.shape[1], I.shape[2]-1)
    D = np.zeros(shape)

    for i in range(1, I.shape[2]):
        D[:,:,i-1] = I[:,:,i] - I[:,:,i-1]

    return D

def check_local_extrema(D, i, j, k):
    target = D[i,j,k]
    for x in [i-1, i, i+1]:
        for y in [j-1, j, j+1]:
            for z in [k-1, k, k+1]:
                if D[x,y,z] > target:
                    return False
    return True

def find_extrema(D, thresh):
    n_scales = D.shape[2]
    height, width = D.shape[:2]
    extrema = []

    for k in range(1, n_scales-1):
        for i in range(1, height-1):
            for j in range(1, width-1):
                isLocalExtrema = check_local_extrema(D, i, j, k)
                # If a point is a local extrema add it to the list
                if isLocalExtrema:
                    # Threshold the point
                    if D[i,j,k] > thresh:
                        extrema.append((i,j,k))

    return extrema

def find_sift_keypoints(local_extrema, octave_num, sigma, n_scales):
    k = 2**(1./(n_scales-1))
    keypoints = []

    for extrema in local_extrema:
        # Upsample the location and scale
        scaling_factor = (2 ** (octave_num-1))

        # i,j coordinate up-scaling
        location = (extrema[0] * scaling_factor + scaling_factor - 1,
                    extrema[1] * scaling_factor + scaling_factor - 1)

        # determining the scale at the keypoint
        scale = (k ** extrema[2]) * sigma * scaling_factor
        sift_keypoint = (location, scale)

        keypoints.append(sift_keypoint)

    return keypoints

# stddev:   standard deviation used for gaussian smoothing of the
#           first scale in each octave
# n_scales: number of scales per octave
# thresh:   threshold for the DoG extrema
def sift_keypoint_detector(image, stddev, n_scales, thresh):
    image_gray = rgb2gray(image)
    sift_keypoints = []
    oct_num = 1

    # COMPUTE GAUSSIAN PYRAMIDS
    pyramid = tuple(pyramid_gaussian(image_gray, multichannel=False))

    # iterate over each octave
    for octave in pyramid:

        # CREATE SCALE SPACE
        I = create_scale_space(octave, sigma=stddev, n_scales=n_scales)

        # COMPUTE DIFFERENCE OF GAUSSIANS
        D = difference_of_gaussians(I)

        # Modify D to make dealing with DoG easier
        D = D**2            # Square to deal with only maxima
        D_max = D.max()     # Scale to 1
        D = D/D_max

        # LOCATE LOCAL EXTREMA
        local_extrema = find_extrema(D, thresh=thresh)

        # STORE LOCAL EXTREMA POSITION AND SCALE IN RETURN LIST
        keypoints = find_sift_keypoints(local_extrema, oct_num, stddev, n_scales)
        sift_keypoints.extend(keypoints)

        oct_num = oct_num + 1

    return sift_keypoints


def main():
    image = io.imread('images/building.jpg', as_gray=False)

    keypoints = sift_keypoint_detector(image,
                                        stddev=1.6,
                                        n_scales=5,
                                        thresh=0.1)

    # Plot detected SIFT keypoints
    for point in keypoints:
        (i,j) = point[0]
        stddev = point[1]
        radius = ceil(2**(0.5) * stddev)

        rr, cc = draw.circle_perimeter(i,j, radius, shape=image.shape)
        image[rr,cc,:] = (255,0,0)


    io.imshow(image)
    io.show()


if __name__ == '__main__':
    main()
