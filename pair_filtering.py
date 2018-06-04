import cv2 as cv
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

from sift_pairs import get_filtered_keypoints_and_descriptions
from utility import append_images


def check_neighbour_integrity(kp_pairs, n, k, verbose=False):
    # kp1, des1, kp2, des2 = get_filtered_keypoints_and_descriptions(img1, img2)
    # kp_pairs = get_filtered_keypoints_and_descriptions(img1, img2)

    kp1 = [pair[0] for pair in kp_pairs]
    kp2 = [pair[1] for pair in kp_pairs]

    # coords1 = list(map(lambda kp: kp.pt, kp1))
    # coords2 = list(map(lambda kp: kp.pt, kp2))
    #
    # print(coords1[0:5])
    # # print(coords)
    #
    # dist_matrix1 = spatial.distance_matrix(coords1, coords1)
    # dist_matrix2 = spatial.distance_matrix(coords2, coords2)
    # # print(np.array(dist_matrix1)[0:7, 0:7])
    #
    # print(dist_matrix1[20])
    # print(np.argsort(dist_matrix1[20]))
    #
    # sorted_kp_indices = []
    #
    # for i in range(len(kp1)):
    #     sorted_kp_indices.append(np.argsort(dist_matrix1[i]))
    #
    # neighbours1 = []
    # for i in range(len(kp1)):
    #     nbs = []
    #     j = 0
    #     while len(nbs) < n:
    #         if coords1[i] != coords1[j]:
    #             nbs.append(kp1[j])
    #         j += 1
    #
    #     neighbours1.append(nbs)
    #
    # for i in range(len(kp1)):
    #     neighbours1 = kp1[np.argsort(dist_matrix1[i])[0:n-1]]

    neighbours1 = get_neighbours_matrix(kp1, n)
    neighbours2 = get_neighbours_matrix(kp2, n)

    # print(spatial.distance.euclidean(neighbours1[0][0].pt, neighbours1[0][2].pt))
    # print()

    valid_neigbours_numbers = []
    for i in range(len(kp_pairs)):
        nbs1 = neighbours1[i]
        nbs2 = neighbours2[i]
        valid_neigbours_number = 0
        for nbr in nbs1:
            nbr_pair = get_pair(nbr, kp_pairs)
            # print(nbr_pair)
            # print(nbs2)
            if nbr_pair in nbs2:
                valid_neigbours_number += 1
        valid_neigbours_numbers.append(valid_neigbours_number)

    valid_pairs = []
    for i in range(len(kp_pairs)):
        if valid_neigbours_numbers[i] >= k:
            valid_pairs.append(kp_pairs[i])

    print('N = ', n)
    print('K = ', k)
    print("All pairs: ", len(kp_pairs))
    print("Valid pairs: ", len(valid_pairs))

    return valid_pairs


class TransformModel:
    def __init__(self, pairs_for_transform, transform):
        if transform == 'affine':


def get_pairs_for_transform(kp_pairs):
    # get 3 pairs
    pairs_for_transform = []

    # totally random
    pairs_for_transform.append(random.choice(kp_pairs))
    pairs_for_transform.append(random.choice([pair for pair in kp_pairs if pair not in pairs_for_transform]))
    pairs_for_transform.append(random.choice([pair for pair in kp_pairs if pair not in pairs_for_transform]))


def calculate_model(pairs_for_transform, transform):
    pass


def ransac(kp_pairs, samples, max_error):

    best_model = None
    best_score = 0

    for i in range(samples):
        pairs_for_transform = get_pairs_for_transform(kp_pairs)

        model = calculate_model(pairs_for_transform, transform='affine')
        score = 0

        for pair in kp_pairs:
            error = model.calculate_error(pair)
            if error < max_error:
                score += 1
        if score > best_score:
            best_score = score
            best_model = model

    verified_pairs = []
    for pair in kp_pairs:
        error = best_model.calculate_error(pair)
        if error < max_error:
            verified_pairs.append(pair)

    return verified_pairs






def get_pair(keypoint, pairs):
    for pair in pairs:
        if pair[0] == keypoint:
            return pair[1]
    return None


def get_neighbours_matrix(kp, n):
    coords = list(map(lambda keypoint: keypoint.pt, kp))

    dist_matrix = spatial.distance_matrix(coords, coords)

    sorted_kp_indices = []

    for i in range(len(kp)):
        sorted_kp_indices.append(np.argsort(dist_matrix[i]))

    neighbours = []
    for i in range(len(kp)):
        nbs = []
        # nbs.append(kp[i])
        j = 0
        while len(nbs) < n:
            if coords[i] != coords[j]:
                nbs.append(kp[sorted_kp_indices[i][j]])
            j += 1

        neighbours.append(nbs)

    # # usun to
    # dist_matrix_sorted = np.sort(dist_matrix, axis=1)
    # print(dist_matrix_sorted[0:10, 0:10])

    return neighbours


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



