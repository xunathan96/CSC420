import numpy as np
import random
INLIER_THRESH = 3

# Parameters:   A list of matching keypoints from both images
#               elements of the list are tuples (keypoint, keypoint)
#               corresponding to a matched pair from both images
def calculate_homography(matched_keypoints):
    n_keypoints = len(matched_keypoints)
    A = np.empty((0,9))

    for t in range(n_keypoints):
        (keypoint_j, keypoint_k) = matched_keypoints[t]
        (x_j, y_j) = keypoint_j.pt
        (x_k, y_k) = keypoint_k.pt

        # CREATE PARTIAL A MATRIX
        a = np.array([
            [x_j, y_j, 1, 0, 0, 0, -x_k*x_j, -x_k*y_j, -x_k],
            [0, 0, 0, x_j, y_j, 1, -y_k*x_j, -y_k*y_j, -y_k]
        ])
        # APPEND PARTIAL TO A
        A = np.append(A, a, axis=0)

    # CALCULATE EIGENVALUES & EIGENVECTORS
    w, v = np.linalg.eig(A.T @ A)

    # CALCULATE HOMOGRAPHY MATRIX H
    i_min = np.argmin(w)
    h = v[:, i_min]
    H = np.reshape(h, (3,3))

    return H

# Parameters:   Homography matrix (H)
#               OpenCV point (pt) of point to transform
def transform_point(H, point):
    (x, y) = point  #.pt
    q = np.array([x, y, 1])

    wp = H @ q
    p = wp/wp[2]

    # Return an OpenCV position tuple (column, row)
    return (p[0], p[1])


def number_of_inliers(H, matched_keypoints):
    n = 0
    for i in range(len(matched_keypoints)):
        (keypoint_j, keypoint_k) = matched_keypoints[i]

        p = np.asarray(keypoint_k.pt)
        p_hat = np.asarray(transform_point(H, keypoint_j.pt))
        d = np.linalg.norm(p - p_hat)

        if d < INLIER_THRESH:
            n = n + 1

    return n

# Parameters:   A list of matched keypoints
# Output:       Best solution for the homography (H)
#               calculated via RANSAC
def RANSAC_homography(matched_keypoints):
    max_inliers = 0
    H_optimal = None

    # ITERATE ~3000 ROUNDS to guarentee 99% accuracy assuming 20% inliers
    for i in range(3000):
        # SELECT 4 RANDOM MATCHES TO CALCULATE HOMOGRAPHY
        matches = random.sample(matched_keypoints, 4)
        H = calculate_homography(matches)

        # CALCULATE NUMBER OF INLIERS
        n_inliers = number_of_inliers(H, matched_keypoints)

        # STORE THE HOMOGRAPHY WITH MOST INLIERS
        if n_inliers > max_inliers:
            max_inliers = n_inliers
            H_optimal = H

    print(max_inliers)
    print(H_optimal)
    return H_optimal
