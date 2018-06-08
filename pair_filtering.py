import cv2 as cv
from scipy import spatial
import numpy as np
import random

from sift_pairs import get_keypoint_pairs
from utility import show_pairs_on_images


class PairFilter:

    def __init__(self, kp_pairs):
        self.kp_pairs = kp_pairs

    def filter_with_neighbourhood_integrity(self, n, k, verbose=False):
        kp_pairs = self.kp_pairs
        kp1 = [pair[0] for pair in kp_pairs]
        kp2 = [pair[1] for pair in kp_pairs]

        neighbours1 = self.get_neighbours_matrix(kp1, n)
        neighbours2 = self.get_neighbours_matrix(kp2, n)

        valid_neigbours_numbers = []
        for i in range(len(kp_pairs)):
            nbs1 = neighbours1[i]
            nbs2 = neighbours2[i]
            valid_neigbours_number = 0
            for nbr in nbs1:
                nbr_pair = self.get_pair(nbr, kp_pairs)
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

    def get_pairs_for_transform(self, transformType='affine'):
        # get 3 pairs
        pairs_for_transform = []

        # totally random
        pairs_for_transform.append(random.choice(self.kp_pairs))
        pairs_for_transform.append(random.choice([pair for pair in self.kp_pairs if pair not in pairs_for_transform]))
        pairs_for_transform.append(random.choice([pair for pair in self.kp_pairs if pair not in pairs_for_transform]))
        if transformType == 'perspective':
            pairs_for_transform.append(
                random.choice([pair for pair in self.kp_pairs if pair not in pairs_for_transform]))

        return pairs_for_transform

    def filter_with_ransac(self, samples, max_error, verbose=False):
        best_model = None
        best_score = 0

        for i in range(samples):
            pairs_for_transform = self.get_pairs_for_transform()

            model = RansacTransformModel(pairs_for_transform, 'affine')
            score = 0

            for pair in self.kp_pairs:
                error = model.calculate_error(pair)
                if error < max_error:
                    score += 1
            if score > best_score:
                best_score = score
                best_model = model

            if verbose:
                print('Iteration ', i, ' - score: ', score)

        verified_pairs = []
        for pair in self.kp_pairs:
            error = best_model.calculate_error(pair)
            if error < max_error:
                verified_pairs.append(pair)

        return verified_pairs

    def get_pair(self, keypoint, pairs):
        for pair in pairs:
            if pair[0] == keypoint:
                return pair[1]
        return None

    def get_neighbours_matrix(self, kp, n):
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

        return neighbours


class RansacTransformModel:
    def __init__(self, pairs_for_transform, transformType='affine'):
        self.model = np.zeros(shape=[3, 3])
        if transformType == 'affine':
            self.create_affine_model(pairs_for_transform)
        if transformType == 'perspective':
            self.create_perspective_model(pairs_for_transform)

    def create_affine_model(self, pairs):
        x1 = pairs[0][0].pt[0]
        x2 = pairs[1][0].pt[0]
        x3 = pairs[2][0].pt[0]

        y1 = pairs[0][0].pt[1]
        y2 = pairs[1][0].pt[1]
        y3 = pairs[2][0].pt[1]

        u1 = pairs[0][1].pt[0]
        u2 = pairs[1][1].pt[0]
        u3 = pairs[2][1].pt[0]

        v1 = pairs[0][1].pt[1]
        v2 = pairs[1][1].pt[1]
        v3 = pairs[2][1].pt[1]

        A = [[x1, y1, 1, 0, 0, 0],
             [x2, y2, 1, 0, 0, 0],
             [x3, y3, 1, 0, 0, 0],
             [0, 0, 0, x1, y1, 1],
             [0, 0, 0, x2, y2, 1],
             [0, 0, 0, x3, y3, 1]]
        A = np.array(A)

        B = [u1, u2, u3, v1, v2, v3]
        B = np.array(B)

        result_vector = np.dot(np.linalg.pinv(A), B)
        result_vector = np.append(result_vector, [0, 0, 1])

        self.model = result_vector.reshape(3, 3)

    def create_perspective_model(self, pairs):
        pass

    def calculate_error(self, pair):
        x = pair[0].pt[0]
        y = pair[0].pt[1]

        u = pair[1].pt[0]
        v = pair[1].pt[1]

        calc = self.model.dot(np.array([x, y, 1]))
        u_calc, v_calc = calc[0], calc[1]

        error = spatial.distance.euclidean([u, v], [u_calc, v_calc])
        return error
