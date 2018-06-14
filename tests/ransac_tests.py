import cv2 as cv
import numpy as np
import random

from sift_pairs import get_keypoint_pairs
from utility import show_pairs_on_images
from pair_filtering import PairFilter


def run_test(name, samples, max_error, transform_type, heuristic=False):

    file1 = f'../images/{name}1.jpg'
    file2 = f'../images/{name}2.jpg'

    img1 = cv.imread(file1)
    img2 = cv.imread(file2)

    kp_pairs = get_keypoint_pairs(img1, img2)

    pairFilter = PairFilter(kp_pairs)

    print(img1.shape)

    img_size = np.max(img1.shape[:1])

    if heuristic:
        min_r = round(0.01 * img_size, 2)
        max_r = round(0.3 * img_size, 2)
    else:
        min_r = 0
        max_r = 0

    # samples = 1000
    # max_error = 20
    # transform_type = 'affine'
    # transform_type = 'perspective'

    valid_pairs, score_history = pairFilter.filter_with_ransac(samples=samples,
                                                               max_error=max_error,
                                                               transform_type=transform_type,
                                                               verbose=True,
                                                               min_r=min_r,
                                                               max_r=max_r)

    print('min_r: ', min_r)
    print('max_r: ', max_r)
    print('Total pairs: ', len(kp_pairs))
    print('Valid pairs: ', len(valid_pairs))
    print('Avg score: ', np.average(score_history))

    transform_type_short = transform_type[0].upper()

    result_filename = f"R_{transform_type_short}_{name}_s{samples}_e{max_error}_m{min_r}_M{max_r}"
    result_suptitle = f"Ransac {transform_type} - {name}\n" \
                      f"samples: {samples}  max_error: {max_error}  min_r: {min_r}  max_r: {max_r}"
    result_title = f"total pairs: {len(kp_pairs)}  valid pairs: {len(valid_pairs)}"

    show_pairs_on_images(file1, file2, valid_pairs, result_filename=result_filename, result_title=result_title, result_suptitle = result_suptitle, show=False)


def run_test_bulk(name, samples_list, max_error_list, transform_type_list, heuristic_list):

    file1 = f'../images/{name}1.jpg'
    file2 = f'../images/{name}2.jpg'

    img1 = cv.imread(file1)
    img2 = cv.imread(file2)

    kp_pairs = get_keypoint_pairs(img1, img2)

    for samples in samples_list:
        print(f"samples: {samples}")
        for max_error in max_error_list:
            for transform_type in transform_type_list:
                for heuristic in heuristic_list:
                    pairFilter = PairFilter(kp_pairs)

                    img_size = np.max(img1.shape[:1])

                    if heuristic:
                        min_r = round(0.01 * img_size, 2)
                        max_r = round(0.3 * img_size, 2)
                    else:
                        min_r = 0
                        max_r = 0

                    valid_pairs, score_history = pairFilter.filter_with_ransac(samples=samples,
                                                                               max_error=max_error,
                                                                               transform_type=transform_type,
                                                                               verbose=False,
                                                                               min_r=min_r,
                                                                               max_r=max_r)

                    # print('min_r: ', min_r)
                    # print('max_r: ', max_r)
                    # print('Total pairs: ', len(kp_pairs))
                    # print('Valid pairs: ', len(valid_pairs))
                    # print('Avg score: ', np.average(score_history))

                    transform_type_short = transform_type[0].upper()

                    result_filename = f"R_{transform_type_short}_{name}_s{samples}_e{max_error}_m{min_r}_M{max_r}"
                    result_suptitle = f"Ransac {transform_type} - {name}\n" \
                                      f"samples: {samples}  max_error: {max_error}  min_r: {min_r}  max_r: {max_r}"
                    result_title = f"total pairs: {len(kp_pairs)}  valid pairs: {len(valid_pairs)}"

                    show_pairs_on_images(file1, file2, valid_pairs, result_filename=result_filename,
                                         result_title=result_title, result_suptitle=result_suptitle, show=False)
