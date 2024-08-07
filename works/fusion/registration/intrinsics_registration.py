# This is the estimation of R and t with K_vi and K_ir, the ca,era calibration
# results.

import numpy as np
from scipy.optimize import least_squares

# import K_vi and K_ir
# wide angle camera intrinsics
K_wide = np.array([
    [2901.19910315714, 0, 940.239619965275],
    [0, 2893.75517626367, 618.475768281058],
    [0, 0, 1]
])

# zoom camera intrinsics
K = np.array([
    [1044.03628677823, 0, 335.125645561794],
    [0, 1051.80215540345, 341.579677246452],
    [0, 0, 1]

# infrared camer aintrinsics
Kir = np.array([
    [1044.03628677823, 0, 335.125645561794],
    [0, 1051.80215540345, 341.579677246452],
    [0, 0, 1]
])



