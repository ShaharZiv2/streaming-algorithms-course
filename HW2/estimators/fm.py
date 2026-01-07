import numpy as np
from .estimator import Estimator
from utils.hash_utils import NormalizedHash
import math

class FMEstimator(Estimator):
    def __init__(self, r: int):
        self.num_estimators = r
        self.num_groups = math.isqrt(r)
        self.estimators = np.ones(r, dtype=float)
        self.hashes = [NormalizedHash() for _ in range(r)]

    def update(self, feature):
        hash_values = np.vectorize(lambda h: h.digest(feature))(self.hashes)

        # If h(a) < X then X = h(a) - vectorized minimum operation
        self.estimators = np.minimum(self.estimators, hash_values)

    def estimate(self) -> int:
        # Reshape into 8 groups of 8 estimators for median-of-means
        groups = self.estimators.reshape(self.num_groups, self.num_groups)
        # Calculate mean estimate for each group
        group_estimates = np.mean((1 / groups) - 1, axis=1)
        # Return median of group estimates
        return int(np.median(group_estimates))

    def report(self):
        """Report memory usage of the FM Estimator."""
        # Memory for the estimators array: r floats (float64)
        estimators_memory = self.estimators.nbytes

        # Memory for hash functions list
        # Each NormalizedHash object is approximately 64 bytes (rough estimate)
        hashes_memory = len(self.hashes) * 64

        # Memory for configuration parameters (num_estimators and num_groups)
        # Python integers typically use 28 bytes each
        config_memory = 2 * 28

        total_memory = estimators_memory + hashes_memory + config_memory

        return total_memory
