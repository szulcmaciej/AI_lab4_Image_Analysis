import cv2 as cv

from sift_pairs import get_keypoint_pairs
from utility import show_pairs_on_images

from pair_filtering import PairFilter


def run_test(name, neighbours_total, neighbours_threshold):
    # file1 = '../images/nutella1.jpg'
    # file2 = '../images/nutella2.jpg'

    # file1 = '../images/room1.png'
    # file2 = '../images/room2.png'

    # file1 = '../images/obrazki1.jpg'
    # file2 = '../images/obrazki2.jpg'

    # name = 'ksiazki'

    file1 = f'../images/{name}1.jpg'
    file2 = f'../images/{name}2.jpg'

    img1 = cv.imread(file1)
    img2 = cv.imread(file2)

    kp_pairs = get_keypoint_pairs(img1, img2)

    # neighbours_total = 6
    # neighbours_threshold = 4

    pairFilter = PairFilter(kp_pairs)
    valid_pairs = pairFilter.filter_with_neighbourhood_integrity(neighbours_total, neighbours_threshold)

    result_filename = f"NA_{name}_n{neighbours_total}_t{neighbours_threshold}"
    result_suptitle = f"Neighbourhood analisys - {name}\n" \
                      f"neighbours: {neighbours_total}  threshold: {neighbours_threshold}"
    result_title = f"total pairs: {len(kp_pairs)}  valid pairs: {len(valid_pairs)}"

    show_pairs_on_images(file1, file2, valid_pairs, result_filename=result_filename, result_title=result_title,
                         result_suptitle=result_suptitle, show=False)

