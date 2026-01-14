import math

import numpy as np

from estimators import Estimator
from utils.hash_utils import SignHash

class AMSEstimator(Estimator):

    def __init__(self, r: int):
        self.sqrt_r = math.isqrt(r)
        self.__init_hash_matrix()
        self.matrix = np.zeros((self.sqrt_r, self.sqrt_r), dtype=int)

    def update(self, feature):
        for row in range(self.sqrt_r):
            for column in range(self.sqrt_r):
                self.matrix[row, column] += self.hash_matrix[row][column].digest(feature)

    def estimate(self):
        squared_estimators = self.matrix ** 2
        return np.median(squared_estimators.mean(axis=1))

    def report(self):
        # I don't really know what to put here....
        pass

    def reset(self):
        self.matrix = np.zeros((self.sqrt_r, self.sqrt_r), dtype=int)

    def __init_hash_matrix(self):
        self.hash_matrix = [[SignHash() for _ in range(self.sqrt_r)] for _ in range(self.sqrt_r)]
