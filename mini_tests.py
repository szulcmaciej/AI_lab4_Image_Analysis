import cv2 as cv
import numpy as np
from scipy import spatial


# des1 = np.array([0,1,2,2,1,0,1,5,0,1]).reshape((2, 5))
# des2 = np.arange(10).reshape((2, 5)) + 1
#
# print(des1)
# print(des2)
#
#
# distances = np.zeros((des1.shape[0], des2.shape[0]))
# for i in range(des1.shape[0]):
#     for j in range(des2.shape[0]):
#         distance = np.sum(np.abs(des1[i] - des2[j]))
#         distances[i, j] = distance
# print(distances)
#
# print()
# print(np.argmin(distances, axis=1))

a = np.array([[0,0], [0,1], [1,1], [1, 0]])
b = np.array([[0,1], [1, 1], [0, 0], [1, 0]])

print(spatial.distance_matrix(a, a))