from collections import defaultdict
from typing import List

import numpy as np
from utils.hash_utils import BucketHash
from estimators.estimator import Estimator


class HeavyHittersSketch(Estimator):
    def __init__(self, num_buckets: int = 5):
        self.num_buckets = num_buckets
        self.buckets = {}

    def update(self, feature):
        if feature in self.buckets:
            self.buckets[feature] += 1
        else:
            if len(self.buckets) < self.num_buckets:
                self.buckets[feature] = 1
            else:
                for heavy_hitter in self.buckets:
                    self.buckets[heavy_hitter] -= 1
                    if self.buckets[heavy_hitter] == 0:
                        self.buckets.pop(heavy_hitter)

    def estimate(self, feature) -> List[str]:
        return list(self.buckets.keys())

    def report(self):
        """Report sketch statistics."""
        pass

    def reset(self):
        self.buckets = {}
