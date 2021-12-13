import math
import numpy as np
from numpy.testing._private.utils import measure
import scipy.linalg

class KalmanFilter(object):
    def __init__(self):
        ndim = 4
        dt = 1

        self._motion_mat = np.eye(2*ndim, 2*ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim +i] = dt
        self._update_mat = np.eye(ndim, 2*ndim)

        # motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in the model.
        self._std_weight_position = 1./ 20
        self._std_weight_velocity = 1./ 160
    def initiate(self, measurement):
        # create track from unassosicated measurement
        '''
        parameters
        ----------
        measurement: ndarray, bounding box with center position (u, v)
        scale (s) and aspect ratio (r)
        ----------
        returns
        ----------
        (ndarray, ndarray) the mean vector (dim = 7) and covariance vector (7 x 7)
        of the new track. Unobserved velocities are initialized to 0 mean.
        '''
        # position
        mean_pos = measurement
        # velocity
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel] # concat matrices by row, T|B, np.c_ --> L|R
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3], 
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        # mean, covariance
        # return: (ndarray, ndarray), the mean vector and covariance matrix of the predicted state
        # position
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
            ]
        # velocity
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]

        motion_cov = np.dot(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        # project state distribution to measurement space
        # mean covariance
        # return: (ndarray, ndarray), returns the project mean and covariance matrix to the given state estimate.
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        ))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        # KF correction
        # mean, covariance, measurement
        # returns: (ndarray, ndarray), the measurement-corrected state distribution
        projected_mean, projected_cov = self.project(mean, covariance)
