import numpy as np
from .estimator import Estimator

class MorrisEstimator(Estimator):
    def __init__(self, num_counters: int):
        self.num_counters = num_counters
        self.counters = np.zeros(num_counters, dtype=int)
        # self.total_updates = 0

    def update(self):
        probabilities = 1 / (2 ** self.counters)
        # Generate random values for all counters at once
        random_values = np.random.random(self.num_counters)
        # Update counters where random value < probability
        self.counters += (random_values < probabilities).astype(int)

    def estimate(self) -> int:
        return int(np.mean(2 ** self.counters - 1))

    def report(self):
        pass

