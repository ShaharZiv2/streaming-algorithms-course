import math

import numpy as np
from utils.hash_utils import SignHash

class AMSEstimator:

    def __init__(self, r: int):
        self.sqrt_r = math.isqrt(r)
        self.__init_hash_matrix()
        self.matrix = np.zeros((self.sqrt_r, self.sqrt_r), dtype=int)

    def update(self, feature):
        # Vectorize the hash computation across the entire matrix
        hash_values = np.vectorize(lambda h: h.digest(feature))(self.hash_matrix)
        self.matrix += hash_values

    def estimate(self):
        squared_estimators = self.matrix ** 2
        return np.median(squared_estimators.mean(axis=1))

    def report(self):
        """Report memory usage of the AMS Estimator."""
        # Memory for the matrix: sqrt(r) x sqrt(r) integers
        matrix_memory = self.matrix.nbytes

        # Memory for hash matrix array structure
        hash_array_memory = self.hash_matrix.nbytes

        # Memory for actual SignHash objects (approximate)
        # Each SignHash object is approximately 64 bytes
        hash_objects_memory = self.sqrt_r * self.sqrt_r * 64

        # Memory for configuration parameter (sqrt_r)
        config_memory = 28

        total_memory = matrix_memory + hash_array_memory + hash_objects_memory + config_memory

        return total_memory

    def __init_hash_matrix(self):
        # Use NumPy array for better memory efficiency
        self.hash_matrix = np.array([[SignHash() for _ in range(self.sqrt_r)]
                                      for _ in range(self.sqrt_r)], dtype=object)
