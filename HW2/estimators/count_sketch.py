import numpy as np
from utils.hash_utils import BucketHash, SignHash
from estimators.estimator import Estimator


class CountSketch(Estimator):
    def __init__(self, r: int, b: int):
        """
        Count Sketch with configurable rows and buckets.

        Args:
            r: Number of rows (typically 5, 7, or 9)
            b: Number of buckets per row (typically 2048 or 8192)
        """
        super().__init__()
        self.rows = r
        self.buckets = b
        self.hash_functions = [BucketHash(b) for _ in range(r)]
        self.sign_functions = [SignHash() for _ in range(r)]
        self.table = np.zeros((r, b), dtype=int)

    def update(self, feature, count: int = 1):
        """Update the sketch with a feature and optional count."""
        for i in range(self.rows):
            bucket = self.hash_functions[i].digest(feature)
            sign = self.sign_functions[i].digest(feature)
            self.table[i][bucket] += sign * count

    def estimate(self, feature) -> int:
        """Estimate the count for a given feature using median of signed estimates."""
        estimates = []
        for i in range(self.rows):
            bucket = self.hash_functions[i].digest(feature)
            sign = self.sign_functions[i].digest(feature)
            estimates.append(sign * self.table[i][bucket])
        return int(np.median(estimates))

    def report(self):
        """Report sketch statistics."""
        pass

