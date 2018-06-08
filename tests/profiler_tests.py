import cv2 as cv
import cProfile

from sift_pairs import get_keypoint_pairs
from pair_filtering import PairFilter


def ransac_test():
    file1 = '../images/nutella1.jpg'
    file2 = '../images/nutella2.jpg'

    # file1 = '../images/room1.png'
    # file2 = '../images/room2.png'

    img1 = cv.imread(file1)
    img2 = cv.imread(file2)

    kp_pairs = get_keypoint_pairs(img1, img2)

    pairFilter = PairFilter(kp_pairs)
    valid_pairs = pairFilter.filter_with_ransac(1000, 50, transform_type='perspective',  verbose=True)


if __name__ == '__main__':
    cProfile.run('ransac_test()', sort='tottime')
