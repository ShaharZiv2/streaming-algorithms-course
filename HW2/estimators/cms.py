import numpy as np
from utils.hash_utils import BucketHash
from estimators.estimator import Estimator


class CountMinSketch(Estimator):
    def __init__(self, width: int):
        """
        Count-Min Sketch with d=4 depth and configurable width w.

        Args:
            w: Width of the sketch table (typically 2048 or 8192)
        """
        self.width = width
        self.depth = 4
        self.hash_functions = [BucketHash(self.width) for _ in range(self.depth)]
        self.table = np.zeros((self.depth, self.width), dtype=int)

    def update(self, feature, count: int = 1):
        """Update the sketch with a feature and optional count."""
        for i in range(self.depth):
            hash_value = self.hash_functions[i].digest(feature)
            self.table[i][hash_value] += count

    def estimate(self, feature) -> int:
        """Estimate the count for a given feature."""
        estimates = [self.table[i][self.hash_functions[i].digest(feature)]
                     for i in range(self.depth)]
        return int(np.min(estimates))

    def report(self):
        """Report sketch statistics."""
        pass

    def reset(self):
        self.table = np.zeros((self.depth, self.width), dtype=int)
