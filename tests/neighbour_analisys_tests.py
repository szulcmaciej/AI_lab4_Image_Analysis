import cv2 as cv

from sift_pairs import get_keypoint_pairs
from utility import show_pairs_on_images

from pair_filtering import PairFilter


if __name__ == '__main__':
    # file1 = '../images/nutella1.jpg'
    # file2 = '../images/nutella2.jpg'

    file1 = '../images/room1.png'
    file2 = '../images/room2.png'

    img1 = cv.imread(file1)
    img2 = cv.imread(file2)

    kp_pairs = get_keypoint_pairs(img1, img2)

    pairFilter = PairFilter(kp_pairs)
    valid_pairs = pairFilter.filter_with_neighbourhood_integrity(6, 4)

    show_pairs_on_images(file1, file2, valid_pairs)
