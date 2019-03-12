from SIFT_matching import *
from homography import *
from scipy import ndimage

# Parameters:   Two images img1, img2
# Output:       List of matching keypoints between the two images
#               elements of the list are in the form
#               (keypoint_1, keypoint_2) where keypoints 1 & 2 are matching
def get_matching_keypoints(img1, img2):
    keypts_1, des_1 = extract_SIFT_keypoints(img1)
    keypts_2, des_2 = extract_SIFT_keypoints(img2)

    matched_keypoints = []

    for i in range(len(keypts_1)):
        # GET THE KEYPOINT MATCH CORRESPONDING TO KEYPOINT i
        f_j, j, thresh_ratio = calculate_correspondance(des_1[i], des_2)

        # THRESHOLD FOR GOOD MATCHES
        if thresh_ratio < 0.8:
            match_tuple = (keypts_1[i], keypts_2[j])
            matched_keypoints.append(match_tuple)

    return matched_keypoints

def stitch_images(H, img1, img2):
    nrow = img1.shape[0]
    ncol = img1.shape[1]
    nrow_2 = img2.shape[0]
    ncol_2 = img2.shape[1]

    # TRANSFORM 4 CORNERS OF FIRST IMAGE
    (topleft_x, topleft_y) = transform_point(H, (0,0))
    (topright_x, topright_y) = transform_point(H, (ncol-1,0))
    (bottomright_x, bottomright_y) = transform_point(H, (ncol-1,nrow-1))
    (bottomleft_x, bottomleft_y) = transform_point(H, (0,nrow-1))

    # CALCULATE DIMENSIONS OF STITCHED IMAGE
    min_y = min(topleft_y, topright_y, 0)
    max_y = max(bottomleft_y, bottomright_y, nrow_2-1)
    min_x = min(topleft_x, bottomleft_x, 0)
    max_x = max(topright_x, bottomright_x, ncol_2-1)
    nrows_stitched = max_y - min_y + 1
    ncols_stitched = max_x - min_x + 1

    # CREATE BLANK STITCHED IMAGE
    shape = (int(nrows_stitched)+1, int(ncols_stitched)+1, 3)
    stitched_image = np.zeros(shape)

    # CALCULATE OFFSET
    offset_x = 0 - min_x
    offset_y = 0 - min_y

    # TRANSFORM ALL PIXELS IN THE FIRST IMAGE
    for x in range(ncol):
        for y in range(nrow):
            (x_t, y_t) = transform_point(H, (x,y))
            x_t = int(round(x_t + offset_x))
            y_t = int(round(y_t + offset_y))
            stitched_image[y_t, x_t, :] = img1[y, x, :]

    # SHIFT ALL PIXELS IN THE SECOND IMAGE
    for x in range(ncol_2):
        for y in range(nrow_2):
            x_t = int(round(x + offset_x))
            y_t = int(round(y + offset_y))
            stitched_image[y_t, x_t, :] = img2[y, x, :]

    # INTERPOLATE SKIPPED PIXELS
    # after applying H some pixels in the stitched image may be skipped
    # b/c float position is converted to int position (during formation of stitched image)
    for x in range(1, stitched_image.shape[1]-1):
        for y in range(1, stitched_image.shape[0]-1):
            # If point is black/skipped then linearly interpolate based on neighbours
            if np.all(stitched_image[y,x,:] == 0):
                interpolate_x = 0.5*(stitched_image[y,x-1,:] + stitched_image[y,x+1,:])
                interpolate_y = 0.5*(stitched_image[y-1,x,:] + stitched_image[y+1,x,:])
                if np.linalg.norm(interpolate_x) > np.linalg.norm(interpolate_y):
                    stitched_image[y,x,:] = interpolate_x
                else:
                    stitched_image[y,x,:] = interpolate_y

    return stitched_image


def main():
    landscape1_path = 'images/landscape_1.jpg'
    landscape2_path = 'images/landscape_2.jpg'
    img1 = cv2.imread(landscape1_path)
    img2 = cv2.imread(landscape2_path)

    # Q3
    matched_keypoints = get_matching_keypoints(img1, img2)
    H = RANSAC_homography(matched_keypoints)
    stitch = stitch_images(H, img1, img2)

    cv2.imwrite('stitch.jpg', stitch)


if __name__ == '__main__':
    main()
