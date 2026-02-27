from logic.constants import COLLECTION_MIN_HASH, COLLECTION_JSONL, SEED
from logic.processing.file_utils import load_jsonl
from retrievers.base_retriever import BaseRetriever
import numpy as np
from datasketch import MinHash
from sklearn.cluster import DBSCAN


class SketchRetriever(BaseRetriever):

    def __init__(self, dbscan_eps: float = 0.5, min_samples: int = 3):
        self.min_hash_docs = None
        self.ids = None
        self.signatures = None
        self.clusters = None
        self.min_hash_cursor = 0

        self.dbscan = DBSCAN(eps=dbscan_eps, min_samples=min_samples, metric='hamming')
        super().__init__()

    def build_corpus(self, num_initial_documents: int = 20000):
        self._set_min_hash_docs()

        self.ids = self.min_hash_docs['doc_ids'][:num_initial_documents]
        self.signatures = self.min_hash_docs['signatures'][:num_initial_documents]
        self.min_hash_cursor = num_initial_documents

        self.clusters = self.dbscan.fit_predict(self.signatures)

    def update(self, num_updates: int = 1):
        self.ids.extend(self.min_hash_docs['doc_ids'][self.min_hash_cursor:num_updates])
        self.signatures.extend(self.min_hash_docs['signatures'][self.min_hash_cursor:num_updates])
        self.min_hash_cursor += num_updates

        self.clusters = self.dbscan.fit_predict(self.signatures)

    def retrieve(self, query):
        pass


    def _set_min_hash_docs(self):
        try:
            self.min_hash_docs = np.load(COLLECTION_MIN_HASH)
        except FileNotFoundError:
            self._generate_min_hash_docs()

    def _generate_min_hash_docs(self):
        keys = []
        sketches = []
        for i, document in enumerate(load_jsonl(COLLECTION_JSONL)):
            print(f"\r{i / 5_000:2f}% done", end="", flush=True)
            key = document["key"]
            data = document["data"]
            m = MinHash(num_perm=128, seed=SEED, gpu_mode="detect")
            for j in range(len(data) - 2):
                m.update(data[j: j + 3].encode("utf-8"))
            keys.append(key)
            sketches.append(m.hashvalues)
        np.savez_compressed(COLLECTION_MIN_HASH, doc_ids=keys, signatures=np.array(sketches))
        self.min_hash_docs = np.load(COLLECTION_MIN_HASH)