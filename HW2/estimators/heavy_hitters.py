from typing import List
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

                [self.buckets.pop(heavy_hitter) for heavy_hitter in list(self.buckets.keys()) if self.buckets[heavy_hitter] == 0]

    def estimate(self) -> List[str]:
        return list(self.buckets.keys())

    def report(self):
        """Report sketch statistics."""
        return self.buckets.__sizeof__()

    def reset(self):
        self.buckets = {}
