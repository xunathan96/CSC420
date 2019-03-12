import numpy as np

A = np.array([
    [1062, 635, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1062, 635, 1, 0, 0, 0],
    [1285, 531, 1, 0, 0, 0, -31097, -12850.2, -24.2],
    [0, 0, 0, 1285, 531, 1, 0, 0, 0],
    [1300, 981, 1, 0, 0, 0, -31460, -23740.2, -24.2],
    [0, 0, 0, 1300, 981, 1, -46150, -34825.5, -35.5],
    [1072, 1072, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1072, 1072, 1, -38056, -38056, -35.5]
])

q1 = np.array([
    1025, 630, 1
])
q2 = np.array([
    1768, 222, 1
])
q3 = np.array([
    1835, 3726, 1
])
q4 = np.array([
    1005, 3494, 1
])

def main():

    # FIND EIGENVALUES AND EIGENVECTORS
    w, v = np.linalg.eig(A.T @ A)

    # CALCULATE HOMOGRAPHY MATRIX H
    i_min_eigenvalue = np.argmin(w)
    h = v[:, i_min_eigenvalue]
    H = np.reshape(h, (3,3))

    # WARP EACH POINT (DOOR CORNERS)
    y = H @ q1
    q1_ = y/y[2]

    y = H @ q2
    q2_ = y/y[2]

    y = H @ q3
    q3_ = y/y[2]

    y = H @ q4
    q4_ = y/y[2]

    # PRINT TRANSFORMED COORDINATES
    print(q1_)
    print(q2_)
    print(q3_)
    print(q4_)

if __name__ == '__main__':
    main()
