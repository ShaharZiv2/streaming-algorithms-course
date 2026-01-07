import numpy as np
from utils.hash_utils import BucketHash
from estimators.estimator import Estimator


class CountMinSketch(Estimator):
    def __init__(self, w: int):
        """
        Count-Min Sketch with d=4 depth and configurable width w.

        Args:
            w: Width of the sketch table (typically 2048 or 8192)
        """
        super().__init__()
        self.width = w
        self.depth = 4
        self.hash_functions = [BucketHash(w) for _ in range(self.depth)]
        self.table = np.zeros((self.depth, w), dtype=int)

    def _compute_hashes(self, feature):
        """Compute hash values for a feature across all hash functions."""
        return np.vectorize(lambda h: h.digest(feature))(self.hash_functions)

    def update(self, feature, count: int = 1):
        """Update the sketch with a feature and optional count."""
        hash_values = self._compute_hashes(feature)
        self.table[np.arange(self.depth), hash_values] += count

    def estimate(self, feature) -> int:
        """Estimate the count for a given feature."""
        hash_values = self._compute_hashes(feature)
        estimates = self.table[np.arange(self.depth), hash_values]
        return int(np.min(estimates))

    def report(self):
        """Report sketch statistics and memory usage."""
        # Memory for the table: depth x width integers
        table_memory = self.table.nbytes

        # Memory for hash functions: depth BucketHash objects
        # Each BucketHash object is approximately 64 bytes (rough estimate)
        hash_memory = self.depth * 64

        # Memory for configuration parameters (width and depth)
        config_memory = 2 * 28

        total_memory = table_memory + hash_memory + config_memory

        return total_memory
