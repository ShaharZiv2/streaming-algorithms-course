import numpy as np
from datasketch import MinHash

from logic.constants import SEED, COLLECTION_JSONL, COLLECTION_MIN_HASH
from logic.processing.file_utils import load_jsonl


def build_min_hash_corpus():
    keys = []
    sketches = []
    for i, document in enumerate(load_jsonl(COLLECTION_JSONL)):
        print(f"\r{i/5_000:2f}% done", end="", flush=True)
        key = document["key"]
        data = document["data"]
        m = MinHash(num_perm=128, seed=SEED, gpu_mode="detect")
        for j in range(len(data) - 2):
            m.update(data[j : j + 3].encode("utf-8"))
        keys.append(key)
        sketches.append(m.hashvalues)
    np.savez_compressed(COLLECTION_MIN_HASH, doc_ids=keys, signatures=np.array(sketches))
