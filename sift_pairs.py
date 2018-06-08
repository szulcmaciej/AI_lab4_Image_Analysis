import cv2 as cv
import numpy as np


def get_filtered_keypoints_and_descriptions(img1, img2):

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

    # print()
    # print(f'Potential pairs number: {len(potential_pairs)}')
    # print(f'Pairs number: {len(pairs)}')

    # print(pairs[0])
    # print(des1[pairs[1][0], :10])
    # print(des2[pairs[1][1], :10])

    # # draw all keypoints
    # img1_with_all_keypoints = cv.drawKeypoints(img1, kp1, img1)
    # cv.imwrite('images/img1_with_all_keypoints.png', img1_with_all_keypoints)
    # img2_with_all_keypoints = cv.drawKeypoints(img2, kp2, img2)
    # cv.imwrite('images/img2_with_all_keypoints.png', img2_with_all_keypoints)

    kp1_filtered = [elem for index, elem in enumerate(kp1) if index in map(lambda x: x[0], pairs)]
    des1_filtered = [elem for index, elem in enumerate(des1) if index in map(lambda x: x[0], pairs)]
    kp2_filtered = [elem for index, elem in enumerate(kp2) if index in map(lambda x: x[1], pairs)]
    des2_filtered = [elem for index, elem in enumerate(des2) if index in map(lambda x: x[1], pairs)]

    # # draw filtered keypoints
    # img1_with_filtered_keypoints = cv.drawKeypoints(img1, kp1_filtered, img1)
    # cv.imwrite('images/img1_with_filtered_keypoints.png', img1_with_filtered_keypoints)
    # img2_with_filtered_keypoints = cv.drawKeypoints(img2, kp2_filtered, img2)
    # cv.imwrite('images/img2_with_filtered_keypoints.png', img2_with_filtered_keypoints)

    # TODO remove duplicates(keypoints with the same coords)

    # return kp1_filtered, des1_filtered, kp2_filtered, des2_filtered
    return kp_pairs


if __name__ == '__main__':
    file1 = 'images/room1.png'
    file2 = 'images/room2.png'

    img1 = cv.imread(file1)
    img2 = cv.imread(file2)

    kp_pairs = get_filtered_keypoints_and_descriptions(img1, img2)
    for pair in kp_pairs:
        print(pair)
