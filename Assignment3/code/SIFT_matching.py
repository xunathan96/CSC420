import cv2
import numpy as np
from skimage import io
from skimage.color import rgb2gray

def extract_SIFT_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    # kp is list of keypoints
    # des is a 2D numpy array with rows as descriptors of keypoints
    return kp, des

def feature_extraction(image_path):
    image = cv2.imread(image_path)
    kp, des = extract_SIFT_keypoints(image)
    image_keypoints = np.copy(image)
    cv2.drawKeypoints(image, kp[:100], image_keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('keypoints.png', image_keypoints)

# Calculate the best feature match for the keypoint with descriptor (f)
# in the list of descriptors (descriptors)
# Returns closest matching descriptor in (descriptors)
# and the threshold ratio of the closest match
def calculate_correspondance(f, descriptors):

    # Calculate the euclidean distance between target_descriptor (f)
    # and all the search descriptors (descriptors)
    similarity = np.linalg.norm(f - descriptors, axis=1)

    # Find the closest and 2nd closest descriptor matches
    i = np.argsort(similarity)[0]   # Closest Match
    f_i = descriptors[i]

    j = np.argsort(similarity)[1]   # 2nd Closest Match
    f_j = descriptors[j]

    # Calculate threshold ratio
    thresh_ratio = similarity[i] / similarity[j]

    # Return the descriptor with the best match, its descriptor
    # index (i), and its threshold ratio
    return f_i, i, thresh_ratio

# Parameters:   two images to find matching keypoints on
# Output:       a list of the 3 best matches and their threshold ratios
#               'best' is defined as the matches with lowest threshold ratios
def get_best_matches(ref, test):
    kp_ref, des_ref = extract_SIFT_keypoints(ref)
    kp_test, des_test = extract_SIFT_keypoints(test)

    s_index_i = np.array([0, 0, 0])     # stores index of ref keypoint
    s_index_j = np.array([0, 0, 0])     # stores index of test keypoint
    s_ratios = np.array([1., 1., 1.])   # stores ratios between the ref and test keypoints

    # Find best matches to keypoint/descriptor (i) in reference image
    for i in range(len(kp_ref)):
        # get the keypoint match in test image corresponding to keypoint i
        f_j, j, thresh_ratio = calculate_correspondance(des_ref[i], des_test)

        # If current threshold ratio is smaller than one of the
        # currently stored ratios then replace that keypoint match
        s = np.argmax(s_ratios)
        if thresh_ratio < s_ratios[s]:
            s_index_i[s] = i
            s_index_j[s] = j
            s_ratios[s] = thresh_ratio

    # create list of keypoint matches in ref and test images
    kp_matches_i = [kp_ref[s_index_i[0]], kp_ref[s_index_i[1]], kp_ref[s_index_i[2]]]
    kp_matches_j = [kp_test[s_index_j[0]], kp_test[s_index_j[1]], kp_test[s_index_j[2]]]
    return kp_matches_i, kp_matches_j, s_ratios


def matching(reference, test):
    ref = cv2.imread(reference)
    test = cv2.imread(test)

    # FIND CLOSEST 3 MATCHES BEWTEEN BOTH IMAGES
    match_i, match_j, ratios = get_best_matches(ref, test)

    # DRAW MATCHED KEYPOINTS
    cv2.drawKeypoints(ref, [match_i[0]], ref, color=(255,0,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(ref, [match_i[1]], ref, color=(0,255,0) ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(ref, [match_i[2]], ref, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.drawKeypoints(test, [match_j[0]], test, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(test, [match_j[1]], test, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(test, [match_j[2]], test, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # OUTPUT IMAGES TO FILE
    cv2.imwrite('ref_keypoints.png', ref)
    cv2.imwrite('test_keypoints.png', test)


def main():
    test_path = 'images/test.png'
    test2_path = 'images/test2.png'
    reference_path = 'images/reference.png'

    # Q1 A)
    feature_extraction(test_path)
    feature_extraction(test2_path)
    feature_extraction(reference_path)

    # Q1 B)
    matching(reference_path, test_path)
    matching(reference_path, test2_path)

if __name__ == '__main__':
    main()
