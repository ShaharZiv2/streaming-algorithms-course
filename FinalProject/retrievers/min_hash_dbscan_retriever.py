from logic.constants import COLLECTION_MIN_HASH, COLLECTION_JSONL, SEED, DBSCAN_MIN_HASH_CORPUS_DIR
from logic.processing.file_utils import load_jsonl

from logic.stemming_utils import stemmed_stop_words, StemmingTokenizer
from retrievers.base_min_hash_dbscan_retriever import BaseMinHashDbscanRetriever
import numpy as np
from datasketch import MinHash
from sklearn.feature_extraction.text import CountVectorizer


class MinHashDbscanRetriever(BaseMinHashDbscanRetriever):

    def __init__(self, dbscan_eps: float = 0.85, corpus_initial_size: int = 100_000):
        self.vectorizer = CountVectorizer(ngram_range=(1, 2),
                                          tokenizer=StemmingTokenizer(),
                                          stop_words=stemmed_stop_words())
        super().__init__(dbscan_eps, corpus_initial_size)

    @property
    def _collection_path(self) -> str:
        return COLLECTION_MIN_HASH

    @property
    def _corpus_dir(self) -> str:
        return DBSCAN_MIN_HASH_CORPUS_DIR

    def retrieve(self, query, **kwargs):
        data = [query]
        try:
            self.vectorizer.fit_transform(data)
        except ValueError:
            return []
        ngrams = self.vectorizer.get_feature_names_out()
        m = MinHash(num_perm=128, seed=SEED, gpu_mode='detect')
        for gram in ngrams:
            m.update(gram.encode('utf-8'))
        query_sig = m.hashvalues

        best_cluster = self._find_best_cluster(query_sig)
        return self._get_cluster_doc_ids(best_cluster)

    def _generate_docs(self):
        keys = []
        sketches = []
        for i, document in enumerate(load_jsonl(COLLECTION_JSONL)):
            print(f'\r{100 * i / self.corpus_size:2f}% done', end='', flush=True)
            key = document['key']
            data = [document['data']]
            try:
                self.vectorizer.fit_transform(data)
            except ValueError:
                continue
            three_grams_data = self.vectorizer.get_feature_names_out()
            m = MinHash(num_perm=128, seed=SEED, gpu_mode='detect')
            for gram in three_grams_data:
                m.update(gram.encode('utf-8'))
            keys.append(key)
            sketches.append(m.hashvalues)
        np.savez_compressed(COLLECTION_MIN_HASH, doc_ids=keys, signatures=np.array(sketches))
        return np.load(COLLECTION_MIN_HASH)

    def build_corpus_from_docs(self, docs: list[dict]):
        """Build the sketch index from a pre-loaded list of {"key":…,"data":…} dicts."""
        print(f"[MinHashRetriever] Building MinHash signatures for {len(docs)} documents…")
        keys = []
        sketches = []
        for doc in docs:
            data = [doc["data"].strip()]
            try:
                self.vectorizer.fit_transform(data)
            except ValueError:
                continue
            ngrams = self.vectorizer.get_feature_names_out()
            m = MinHash(num_perm=128, seed=SEED, gpu_mode="detect")
            for gram in ngrams:
                m.update(gram.encode("utf-8"))
            keys.append(doc["key"])
            sketches.append(m.hashvalues)

        self.ids = keys
        self.signatures = np.array(sketches)
        self.corpus_size = len(keys)

        print(f"[MinHashRetriever] Running DBSCAN clustering…")
        min_samples = max(2, self.corpus_size // 100_000) if self.corpus_size >= 2 else 1
        self.dbscan.set_params(min_samples=min_samples)
        self.clusters = self._predict_clusters(self.signatures)

        import pandas as pd
        self.corpus_df = pd.DataFrame({"doc_id": self.ids, "cluster": self.clusters})
        self._compute_centroids()
        self.min_hash_cursor = len(keys)
        print(f"[MinHashRetriever] Index ready ({len(keys)} docs, "
              f"{len(self.centroids)} clusters).")

    def update(self, document: dict | int = 1):
        """Append a single document and re-cluster.

        Accepts either a {"key":…,"data":…} dict (benchmark streaming interface)
        or the legacy integer form.
        """
        if isinstance(document, dict):
            data = [document["data"].strip()]
            try:
                self.vectorizer.fit_transform(data)
            except ValueError:
                return
            ngrams = self.vectorizer.get_feature_names_out()
            m = MinHash(num_perm=128, seed=SEED, gpu_mode="detect")
            for gram in ngrams:
                m.update(gram.encode("utf-8"))
            self.ids.append(document["key"])
            new_sig = m.hashvalues.reshape(1, -1)
            self.signatures = np.vstack([self.signatures, new_sig])
        else:
            # legacy integer path
            n = document
            self.ids.extend(self.docs['doc_ids'][self.min_hash_cursor:self.min_hash_cursor + n])
            new_sigs = self.docs['signatures'][self.min_hash_cursor:self.min_hash_cursor + n]
            self.signatures = np.vstack([self.signatures, new_sigs])
            self.min_hash_cursor += n
            self.corpus_size += n

        self.clusters = self._predict_clusters(self.signatures)
        import pandas as pd
        self.corpus_df = pd.DataFrame({"doc_id": self.ids, "cluster": self.clusters})
        self._compute_centroids()
