import random
from abc import ABC, abstractmethod

import xxhash


class Hash(ABC):

    def __init__(self):
        self.seed = random.getrandbits(128)

    def base_digest(self, feature) -> int:
        return xxhash.xxh64(feature.encode('utf-8'), self.seed).intdigest()


    @abstractmethod
    def digest(self, feature) -> int:
        pass

class NormalizedHash(Hash):

    def digest(self, feature) -> int:
        return self.base_digest(feature) / (2**64)

class BucketHash(Hash):

    def __init__(self, num_buckets):
        super().__init__()
        self.num_buckets = num_buckets

    def digest(self, feature) -> int:
        return self.base_digest(feature) % self.num_buckets


class SignHash(Hash):

    def digest(self, feature) -> int:
        # if even return -1, if odd return 1
        return self.base_digest(feature) % 2 or -1
