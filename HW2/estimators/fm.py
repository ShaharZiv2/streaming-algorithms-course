import numpy as np
from .estimator import Estimator
from utils.hash_utils import NormalizedHash

class FMEstimator(Estimator):
    def __init__(self, num_estimators: int):
        self.num_estimators = num_estimators
        self.estimators = np.ones(num_estimators, dtype=float)
        self.hash = NormalizedHash()

    def update(self, feature):
        hash_value = self.hash.digest(feature)
        hash_values = np.full(self.num_estimators, hash_value)

        # If h(a) < X then X = h(a) - vectorized minimum operation
        self.estimators = np.minimum(self.estimators, hash_values)

    def estimate(self) -> int:
        Z = np.mean(self.estimators)
        return int((1 / Z) - 1)

    def report(self):
        pass
