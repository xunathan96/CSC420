import numpy as np
from skimage import io
from skimage.draw import polygon_perimeter
from skimage.color import rgb2gray
from scipy import ndimage
import scipy.io as sio
import cv2
from depth import *

# Parameters: dets
#   a dictionary of 'car', 'person', and 'bicycle' detections
#   with each row in the value matrix being [xleft,ytop,xright,ybottom,id,score]
# Return: detections
#   a dictionary of all detections
#   key: 'car', 'person', or 'bicycle'
#   value: (top_left_point, bottom_right_point) of detected object
def get_detections(dets):
    i = 0
    detections = {'car': [], 'person': [], 'bicycle': []}

    # order is always car, person, bicycle
    for obj in ['car', 'person', 'bicycle']:
        DS = dets[i][0]
        n_detections = DS.shape[0]

        if n_detections != 0:
            for j in range(n_detections):
                top_left = (DS[j,0], DS[j,1])
                bottom_right = (DS[j,2], DS[j,3])
                box = (top_left, bottom_right)
                detections[obj].append(box)
        i=i+1

    return detections

# Parameters: detections, img
#   detections: a dictionary of bounded boxes around detected objects
#               key: 'car', 'person', 'bicycle'
#               value: (top_left_point, bottom_right_point) of detected object
#   img: the image that detected objects are from
# Return:
#   a image with bounded boxes drawn and labels displayed
def outline_detected_objects(detections, img):
    for obj, detect_list in detections.items():
        for i in range(len(detect_list)):
            (top_left, bottom_right) = detect_list[i]
            # cast to int because opencv draw only accepts ints
            top_left = (int(top_left[0])+1, int(top_left[1])+1)
            bottom_right = (int(bottom_right[0])+1, int(bottom_right[1])+1)

            if obj=='car':
                color = (0,0,255)   # red in opencv
            elif obj=='person':
                color = (255,0,0)   # blue in opencv
            elif obj=='bicycle':
                color = (0,255,0)   # green in opencv (I choose to use green instead of cyan)

            cv2.rectangle(img, top_left, bottom_right, color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text_pos = (top_left[0], top_left[1]+10)
            cv2.putText(img, obj, text_pos, font, 0.65, (255,255,255), 2, cv2.LINE_AA)

    return img

# p_img: point on the image plane
# returns the 3D point X, Y, Z
def calculate_3D_positon(depth_map, p_img):
    x_img = p_img[0]
    y_img = p_img[1]

    # calculate conversion from pixels to mm
    # = px / (width of image * 0.5)
    mm_per_pixel_y = py / (depth_map.shape[0]/2)
    mm_per_pixel_x = px / (depth_map.shape[1]/2)

    # get depth at point (x, y)
    # -- recall that indexing using opencv coordinates is (y, x)
    Z = depth_map[y_img, x_img]
    X = Z * (x_img * mm_per_pixel_x - px) / f    # recall px, py, f are globals
    Y = Z * (y_img * mm_per_pixel_y - py) / f

    return (X, Y, Z)

# Parameters:
#       depth_map: the depth map of the image
#       detections: dictionary of object detections for keys: 'car', 'person', 'bicycle'
#                   the values are lists of (top_left, bottom_right) tuples of the bounded box
#                   around the detected object
# Return: a dictionary of the (X, Y, Z) positions of all detected objects
def calculate_object_positions(depth_map, detections):
    obj_positions = {'car': [], 'person': [], 'bicycle': []}
    for obj, detect_list in detections.items():
        for i in range(len(detect_list)):
            (top_left, bottom_right) = detect_list[i]
            x_left = top_left[0]; y_top = top_left[1]
            x_right = bottom_right[0]; y_bottom = bottom_right[1]

            # Calculate the center of the detected object
            obj_x = (x_left + x_right)/2
            obj_y = (y_top + y_bottom)/2
            p_img = (int(round(obj_x)), int(round(obj_y)))

            # Calculate the 3D position
            p_3D = calculate_3D_positon(depth_map, p_img)
            obj_positions[obj].append(p_3D)

    return obj_positions


def main():
    # get images
    img1_path = 'data/test/left/004945.jpg'
    img2_path = 'data/test/left/004964.jpg'
    img3_path = 'data/test/left/005002.jpg'
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img3 = cv2.imread(img3_path)

    # get disparity maps
    img1_dis_path = 'data/test/results/004945_left_disparity.png'
    img2_dis_path = 'data/test/results/004964_left_disparity.png'
    img3_dis_path = 'data/test/results/005002_left_disparity.png'
    disparity_map1 = cv2.imread(img1_dis_path, cv2.IMREAD_GRAYSCALE).astype(float)
    disparity_map2 = cv2.imread(img2_dis_path, cv2.IMREAD_GRAYSCALE).astype(float)
    disparity_map3 = cv2.imread(img3_dis_path, cv2.IMREAD_GRAYSCALE).astype(float)

    # get matlab data
    img1_mat = 'data/test/results/dets-test/004945_dets.mat'
    img2_mat = 'data/test/results/dets-test/004964_dets.mat'
    img3_mat = 'data/test/results/dets-test/005002_dets.mat'
    detect_results1 = sio.loadmat(img1_mat)['dets']
    detect_results2 = sio.loadmat(img2_mat)['dets']
    detect_results3 = sio.loadmat(img3_mat)['dets']

    # DETECT AND VISUALIZE OBJECTS (car, person, bicycle)
    detections1 = get_detections(detect_results1)
    detections2 = get_detections(detect_results2)
    detections3 = get_detections(detect_results3)
    img_detections_1 = outline_detected_objects(detections1, img1)
    img_detections_2 = outline_detected_objects(detections2, img2)
    img_detections_3 = outline_detected_objects(detections3, img3)

    cv2.imwrite('004945-detections.png', img_detections_1)
    cv2.imwrite('004964-detections.png', img_detections_2)
    cv2.imwrite('005002-detections.png', img_detections_3)

    # COMPUTE THE 3D LOCATION OF EACH DETECTED OBJECT
    depth_map1 = depth_map(disparity_map1)
    depth_map2 = depth_map(disparity_map2)
    depth_map3 = depth_map(disparity_map3)

    img1_positions = calculate_object_positions(depth_map1, detections1)
    img2_positions = calculate_object_positions(depth_map2, detections2)
    img3_positions = calculate_object_positions(depth_map3, detections3)

    print(img1_positions)
    print(img2_positions)
    print(img3_positions)


if __name__ == '__main__':
    main()
