import numpy as np
from .estimator import Estimator
import math

class MorrisEstimator(Estimator):
    def __init__(self, r: int):
        self.num_counters = r
        self.num_groups =  math.isqrt(r)
        self.counters = np.zeros(r, dtype=int)

    def update(self):
        probabilities = 1 / (2 ** self.counters)
        # Generate random values for all counters at once
        random_values = np.random.random(self.num_counters)
        # Update counters where random value < probability
        self.counters += (random_values < probabilities).astype(int)

    def estimate(self) -> int:
        groups = self.counters.reshape(self.num_groups, self.num_groups)
        group_means = np.mean(2 ** groups - 1, axis=1)
        return int(np.median(group_means))

    def report(self):
        """Report memory usage of the Morris Estimator."""
        # Memory for the counters array: r integers
        # numpy int uses platform-dependent size, but counters are dtype=int (typically int64)
        counters_memory = self.counters.nbytes

        # Memory for configuration parameters (num_counters and num_groups)
        # Python integers typically use 28 bytes each
        config_memory = 1 * 28

        total_memory = counters_memory + config_memory

        return total_memory

