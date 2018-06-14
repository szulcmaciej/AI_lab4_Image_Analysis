from tests import ransac_tests, neighbour_analisys_tests

names = ['dali', 'katedra', 'ksiazki', 'muzeum', 'nutella', 'obrazki', 'palac', 'ratusz', 'sciana']

neighbours_thresholds = [(3, 0), (3, 1), (3, 2), (3, 3), (6,4), (10,7)]

# for name in names:
#     print(name)
#     for nt in neighbours_thresholds:
#         print(f"NA {nt[1]}/{nt[0]}")
#         neighbour_analisys_tests.run_test(name, nt[0], nt[1])


ransac_samples = [10, 100, 1000]
transform_types = ['affine', 'perspective']
max_errors = [1, 3, 10, 30, 100]
heuristic_list = [False]

for name in names:
    print(name)
    ransac_tests.run_test_bulk(name, ransac_samples, max_errors, transform_types, heuristic_list)
