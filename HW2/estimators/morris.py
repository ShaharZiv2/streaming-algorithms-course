import numpy as np
from .estimator import Estimator
import math

class MorrisEstimator(Estimator):
    def __init__(self, r: int = 64):
        super().__init__()
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
        pass

    def reset(self):
        self.counters.fill(0)

    def vectorized_update(self, vector):
        for _ in range(len(vector)):
            self.update()

