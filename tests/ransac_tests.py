import cv2 as cv
import numpy as np

from sift_pairs import get_keypoint_pairs
from utility import show_pairs_on_images
from pair_filtering import PairFilter


if __name__ == '__main__':
    # file1 = '../images/nutella1.jpg'
    # file2 = '../images/nutella3.jpg'

    file1 = '../images/room1.png'
    file2 = '../images/room2.png'

    img1 = cv.imread(file1)
    img2 = cv.imread(file2)

    kp_pairs = get_keypoint_pairs(img1, img2)

    pairFilter = PairFilter(kp_pairs)

    print(img1.shape)

    # print('size: ', width, ' x ', height)

    img_size = np.average(img1.shape[:1])

    min_r = 0.01 * img_size
    max_r = 0.3 * img_size

    # print(min_r)
    # print(max_r)

    min_r = 0
    max_r = 0

    valid_pairs, score_history = pairFilter.filter_with_ransac(300, 50, transform_type='affine',  verbose=True, min_r=min_r, max_r=max_r)

    print('Total pairs: ', len(kp_pairs))
    print('Valid pairs: ', len(valid_pairs))
    print('Avg score: ', np.average(score_history))

    show_pairs_on_images(file1, file2, valid_pairs)
    # show_pairs_on_images(file1, file2, valid_pairs[:int(0.1 * len(valid_pairs))])
