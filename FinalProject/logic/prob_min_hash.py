import numpy as np
import xxhash
from datasketch import MinHash

from logic.constants import SEED


class ProbMinHash4:

    def __init__(self, num_perm: int = 128, seed: int = SEED):
        self.num_perm = num_perm
        self.seed = seed
        self.hashvalues = None

        i = np.arange(num_perm - 1)

        raw_boundaries = np.log1p((i + 1) / (num_perm - i - 1))
        self.first_boundary = raw_boundaries[0]
        self.first_boundary_inv = 1.0 / self.first_boundary

        self.boundaries = raw_boundaries / self.first_boundary
        self.trunc_exp_limits = np.diff(raw_boundaries, prepend=0.0)

    def fit(self, keys, weights):
        m = self.num_perm
        result = [None] * m

        q = np.full(m, np.inf)
        max_q = np.inf
        max_q_index = 0

        for key, weight in zip(keys, weights):
            if weight <= 0:
                continue

            w_inv = 1.0 / weight

            seed = xxhash.xxh64(key.tobytes(), seed=self.seed).intdigest()
            rng = np.random.default_rng(seed)

            perm_map = {}

            h = w_inv * self._get_trunc_exp(rng, self.trunc_exp_limits[0])
            i = 1

            while h < max_q:
                swap_index = rng.integers(i - 1, m)

                val_current = perm_map.get(i - 1, i - 1)
                val_swap = perm_map.get(swap_index, swap_index)

                perm_map[i - 1] = val_swap
                perm_map[swap_index] = val_current
                k = val_swap

                if h < q[k]:
                    q[k] = h
                    result[k] = key

                    if k == max_q_index:
                        max_q_index = int(np.argmax(q))
                        max_q = q[max_q_index]

                if not (w_inv * self.boundaries[i - 1] < max_q):
                    break

                if i < m - 1:
                    step = self._get_trunc_exp(rng, self.trunc_exp_limits[i]) / self.first_boundary
                    h = w_inv * (self.boundaries[i - 1] + step)
                else:
                    h = w_inv * (self.boundaries[m - 2] + self.first_boundary_inv * rng.exponential())
                    if h < max_q:

                        k = perm_map.get(m - 1, m - 1)

                        if h < q[k]:
                            q[k] = h
                            result[k] = key
                            if k == max_q_index:
                                max_q_index = int(np.argmax(q))
                                max_q = q[max_q_index]
                    break

                i += 1

        self.hashvalues = result

    def to_min_hash(self) -> MinHash:
        return MinHash(num_perm=self.num_perm, hashvalues=self.hashvalues)

    @staticmethod
    def _get_trunc_exp(rng, limit):
        u = rng.uniform(0, 1)
        return -np.log(1.0 - u * (1.0 - np.exp(-limit)))