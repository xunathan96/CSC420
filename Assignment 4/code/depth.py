import numpy as np
from skimage import io
from skimage.color import rgb2gray
from scipy import ndimage

'''
004945.jpg
    f: 721.537700
    px: 609.559300
    py: 172.854000
    baseline: 0.5327119288

004964.jpg
    f: 721.537700
    px: 609.559300
    py: 172.854000
    baseline: 0.5327119288

005002.jpg
    f: 721.537700
    px: 609.559300
    py: 172.854000
    baseline: 0.5327119288
'''
# CAMERA INTRINSICS (in mm)
f = 721.5377
px = 609.5593
py = 172.854
T = 532.7119288

def depth_map(disparity_map):
    disparity_map[disparity_map<=0] = 1e-2 # approximate 0
    depth_map = f*T/(disparity_map)
    return depth_map

def main():
    img1_path = 'data/test/results/004945_left_disparity.png'
    img2_path = 'data/test/results/004964_left_disparity.png'
    img3_path = 'data/test/results/005002_left_disparity.png'
    disp_img1 = io.imread(img1_path, as_gray=True).astype(float)
    disp_img2 = io.imread(img2_path, as_gray=True).astype(float)
    disp_img3 = io.imread(img3_path, as_gray=True).astype(float)

    # apply gaussian blur to remove noise in disparity map
    # if we dont do this then we can not visualize the map b/c the noise
    # incorrectly scales the visualized image
    disp_img1 = ndimage.gaussian_filter(disp_img1, sigma=2)
    disp_img2 = ndimage.gaussian_filter(disp_img2, sigma=5)
    disp_img3 = ndimage.gaussian_filter(disp_img3, sigma=6)

    depth_map1 = depth_map(disp_img1)
    depth_map2 = depth_map(disp_img2)
    depth_map3 = depth_map(disp_img3)

    # normalize depth max for display only
    max = np.max(depth_map1)
    depth_map1 = depth_map1/max
    max = np.max(depth_map2)
    depth_map2 = depth_map2/max
    max = np.max(depth_map3)
    depth_map3 = depth_map3/max

    io.imsave('004945 - depth_map.png', depth_map1)
    io.imsave('004964 - depth_map.png', depth_map2)
    io.imsave('005002 - depth_map.png', depth_map3)


if __name__ == '__main__':
    main()
