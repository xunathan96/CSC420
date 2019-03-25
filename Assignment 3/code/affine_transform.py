from SIFT_matching import *

def calculate_affine_transform(img1, img2):
    # FIND CLOSEST 3 MATCHES BEWTEEN BOTH IMAGES
    match_i, match_j, ratios = get_best_matches(img1, img2)
    n_keypoints = ratios.shape[0]

    P = np.empty((0,6))
    P_prime = np.empty((0))

    for k in range(n_keypoints):
        # GET KEYPOINT POSITIONS
        kp_i = match_i[k]
        kp_j = match_j[k]
        x_i = kp_i.pt[0]; y_i = kp_i.pt[1]
        x_j = kp_j.pt[0]; y_j = kp_j.pt[1]

        # CREATE PARTIAL P MATRIX
        p = np.array([
            [x_i, y_i, 0, 0, 1, 0],
            [0, 0, x_i, y_i, 0, 1]
        ])
        p_prime = np.array([
            x_j, y_j
        ])

        # APPEND TO P
        P = np.append(P, p, axis=0)
        P_prime = np.append(P_prime, p_prime)

    # CALCULATE AFFINE TRANSFORM MATRIX
    a = np.linalg.inv(P.T @ P) @ P.T @ P_prime
    A = np.array([
        [a[0], a[1], a[4]],
        [a[2], a[3], a[5]]
    ])

    return A

def visualize_affine_transform(ref, test):

    # GET AFFINE TRANSFORM
    A = calculate_affine_transform(ref, test)

    # GET REFERENCE CORNER POINTS (using OpenCV Point coordinate system)
    # in OpenCV Point(x,y) is (column,row) ... very confusing...
    x_max = ref.shape[1]-1
    y_max = ref.shape[0]-1
    p1 = np.array([0, 0, 1])
    p2 = np.array([x_max, 0, 1])
    p3 = np.array([x_max, y_max, 1])
    p4 = np.array([0, y_max, 1])

    # APPLY AFFINE TRANSFORM
    t1 = A @ p1
    t2 = A @ p2
    t3 = A @ p3
    t4 = A @ p4
    # round and cast to int b/c cv2.line requires int datatypes
    t1 = np.around(t1).astype(int)
    t2 = np.around(t2).astype(int)
    t3 = np.around(t3).astype(int)
    t4 = np.around(t4).astype(int)

    # PLOT TRANSFORMED POINTS
    cv2.line(test, tuple(t1), tuple(t2), (255,0,0), 3)
    cv2.line(test, tuple(t2), tuple(t3), (255,0,0), 3)
    cv2.line(test, tuple(t3), tuple(t4), (255,0,0), 3)
    cv2.line(test, tuple(t4), tuple(t1), (255,0,0), 3)

    cv2.imwrite('affine_transform_test.png', test)


def main():
    test_path = 'images/test.png'
    test2_path = 'images/test2.png'
    reference_path = 'images/reference.png'

    ref = cv2.imread(reference_path)
    test = cv2.imread(test_path)
    test2 = cv2.imread(test2_path)

    # Q1 C)
    calculate_affine_transform(ref, test)

    # Q1 D)
    visualize_affine_transform(ref, test)
    visualize_affine_transform(ref, test2)

if __name__ == '__main__':
    main()
