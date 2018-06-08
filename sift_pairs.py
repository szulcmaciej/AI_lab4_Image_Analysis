import cv2 as cv
import numpy as np


def get_keypoint_pairs(img1, img2):

    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    distances = np.zeros((des1.shape[0], des2.shape[0]))
    for i in range(des1.shape[0]):
        for j in range(des2.shape[0]):
            distance = np.sum(np.abs(des1[i] - des2[j]))
            distances[i, j] = distance
    # print(np.argmin(distances, axis=1))

    potential_pairs = [(p1, np.argmin(distances[p1])) for p1 in range(distances.shape[0])]
    pairs = []
    for pair in potential_pairs:
        if np.argmin(distances[:, pair[1]]) == pair[0]:
            pairs.append(pair)

    kp_pairs = [(kp1[pair[0]], kp2[pair[1]]) for pair in pairs]

    # removing points with the same coords
    coord_pairs = [((pair[0].pt, pair[1].pt), pair) for pair in kp_pairs]

    used_coords = []
    verified_pairs = []
    for coords, pair in coord_pairs:
        if coords not in used_coords:
            used_coords.append(coords)
            verified_pairs.append(pair)

    print('kp_pairs: ', len(kp_pairs))
    print('verified pairs (no doubles): ', len(verified_pairs))

    # return kp1_filtered, des1_filtered, kp2_filtered, des2_filtered
    return verified_pairs


if __name__ == '__main__':
    file1 = 'images/room1.png'
    file2 = 'images/room2.png'

    img1 = cv.imread(file1)
    img2 = cv.imread(file2)

    kp_pairs = get_keypoint_pairs(img1, img2)
    # for pair in kp_pairs:
    #     print(pair[0].pt, '    ', pair[1].pt)
