import cv2 as cv
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sift_pairs import get_filtered_keypoints_and_descriptions
from utility import append_images

import pair_filtering


if __name__ == '__main__':
    file1 = 'images/nutella1.jpg'
    file2 = 'images/nutella2.jpg'

    # file1 = 'images/room1.png'
    # file2 = 'images/room2.png'

    img1 = cv.imread(file1)
    img2 = cv.imread(file2)

    kp_pairs = get_filtered_keypoints_and_descriptions(img1, img2)

    valid_pairs = check_neighbour_integrity(kp_pairs, 6, 4)


    # # draw valid keypoints
    # kp1 = [pair[0] for pair in valid_pairs]
    # kp2 = [pair[1] for pair in valid_pairs]
    # img1_with_all_keypoints = cv.drawKeypoints(img1, kp1, img1)
    # cv.imwrite('images/img1_with_valid_keypoints.png', img1_with_all_keypoints)
    # img2_with_all_keypoints = cv.drawKeypoints(img2, kp2, img2)
    # cv.imwrite('images/img2_with_valid_keypoints.png', img2_with_all_keypoints)

    # print(valid_pairs)

    width1 = img1.shape[0]
    width2 = img2.shape[0]
    height1 = img2.shape[1]
    height2 = img2.shape[1]
    #
    images = list(map(Image.open, [file1, file2]))

    # new_im = concat_images(img1, img2)
    new_im = append_images(images, aligment='top')

    x_offset = img1.shape[1]
    for pair in valid_pairs:
        plt.plot([pair[0].pt[0], pair[1].pt[0] + x_offset], [pair[0].pt[1], pair[1].pt[1]], '#FFFF0033')
        # plt.plot([pair[0].pt[0], pair[1].pt[0] + x_offset], [pair[0].pt[1], pair[1].pt[1]])
        # plt.plot([0, 100], [100, 200], 'k-')

    # for pair in valid_pairs:
    #     print(pair[0].pt[1])

    plt.imshow(new_im)
    plt.show()
