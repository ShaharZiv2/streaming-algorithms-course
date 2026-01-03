import random
import xxhash


class Hash:

    def __init__(self, num_values: int):
        self.seed = random.getrandbits(128)
        self.num_values = num_values

    def digest(self, feature):
        if isinstance(feature, str):
            encoded_feature = feature.encode('utf-8')
        else:
            encoded_feature = bytes(feature)

        return xxhash.xxh64(encoded_feature, self.seed).intdigest() % self.num_values
