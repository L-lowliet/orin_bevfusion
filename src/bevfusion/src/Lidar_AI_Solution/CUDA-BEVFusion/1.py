# -*- coding: utf-8 -*-
import numpy as np

# 给定矩阵
matrix = np.array([
    [1.000, 0.007, -0.004, -0.013],
    [0.003, 0.019, 1.000, 0.765],
    [0.007, -1.000, 0.019, -0.311],
    [0.000, 0.000, 0.000, 1.000]
])
# matrix = np.array([
#     [1.000, 0.007, -0.004],
#     [0.003, 0.019, 1.000,],
#     [0.007, -1.000, 0.019]
# ])
# 计算矩阵的逆
inverse_matrix = np.linalg.inv(matrix)
print(inverse_matrix)