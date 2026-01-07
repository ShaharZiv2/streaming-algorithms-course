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

    def _compute_hashes(self, feature):
        """Compute bucket and sign hashes for a feature."""
        buckets = np.vectorize(lambda h: h.digest(feature))(self.hash_functions)
        signs = np.vectorize(lambda s: s.digest(feature))(self.sign_functions)
        return buckets, signs

    def update(self, feature, count: int = 1):
        """Update the sketch with a feature and optional count."""
        buckets, signs = self._compute_hashes(feature)
        self.table[np.arange(self.rows), buckets] += signs * count

    def estimate(self, feature) -> int:
        """Estimate the count for a given feature using median of signed estimates."""
        buckets, signs = self._compute_hashes(feature)
        estimates = signs * self.table[np.arange(self.rows), buckets]
        return int(np.median(estimates))

    def report(self):
        """Report sketch statistics and memory usage."""
        # Memory for the table: rows x buckets integers
        table_memory = self.table.nbytes

        # Memory for hash functions: rows BucketHash objects + rows SignHash objects
        # Each hash object is approximately 64 bytes (rough estimate)
        hash_memory = (len(self.hash_functions) + len(self.sign_functions)) * 64

        # Memory for configuration parameters (rows and buckets)
        config_memory = 2 * 28

        total_memory = table_memory + hash_memory + config_memory

        return total_memory

