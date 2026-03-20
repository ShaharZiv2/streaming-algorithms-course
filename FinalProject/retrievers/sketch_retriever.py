from logic.constants import COLLECTION_MIN_HASH, COLLECTION_JSONL, SEED, CORPUS_DIR
from logic.processing.file_utils import load_jsonl

from logic.stemming_utils import stemmed_stop_words, StemmingTokenizer
from retrievers.base_retriever import BaseRetriever
import numpy as np
import pandas as pd
from datasketch import MinHash
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import CountVectorizer


class SketchRetriever(BaseRetriever):

    def __init__(self, dbscan_eps: float = 0.85, num_initial_documents: int = 100_000):
        self.min_hash_docs = None
        self.ids = None
        self.signatures = None
        self.clusters = None
        self.centroids = {}
        self.min_hash_cursor = 0
        self.num_initial_documents = num_initial_documents

        min_samples = max(2, num_initial_documents // 100_000)
        self.dbscan = DBSCAN(eps=dbscan_eps, min_samples=min_samples, metric='hamming', n_jobs=-1)
        self.vectorizer = CountVectorizer(ngram_range=(1, 2),
                                          tokenizer=StemmingTokenizer(),
                                          stop_words=stemmed_stop_words(),)
        super().__init__(lazy=(num_initial_documents == 0))

    def build_corpus(self):

        self._set_min_hash_docs()
        self.ids = self.min_hash_docs['doc_ids'][:self.num_initial_documents]
        self.signatures = self.min_hash_docs['signatures'][:self.num_initial_documents]
        self.min_hash_cursor = self.num_initial_documents

        try:
            self.clusters = np.load(f'{CORPUS_DIR}/{self.num_initial_documents}_{self.dbscan.eps}.npy')
        except FileNotFoundError:
            self.clusters = self.dbscan.fit_predict(self.signatures)
            np.save(f'{CORPUS_DIR}/{self.num_initial_documents}_{self.dbscan.eps}', arr=self.clusters)

        self.corpus_df = pd.DataFrame({'doc_id': self.ids, 'cluster': self.clusters})

        self.centroids = {}
        for cluster_label in np.unique(self.clusters):
            if cluster_label == -1:
                continue
            mask = self.clusters == cluster_label
            mean_signature = np.mean(self.signatures[mask], axis=0)
            self.centroids[cluster_label] = mean_signature

    def build_corpus_from_docs(self, docs: list[dict]):
        """Build the sketch index from a pre-loaded list of {"key":…,"data":…} dicts."""
        print(f"[SketchRetriever] Building MinHash signatures for {len(docs)} documents…")
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

        print(f"[SketchRetriever] Running DBSCAN clustering…")
        min_samples = max(2, len(keys) // 100_000) if len(keys) >= 2 else 1
        self.dbscan = DBSCAN(
            eps=self.dbscan.eps, min_samples=min_samples,
            metric="hamming", n_jobs=-1,
        )
        self.clusters = self.dbscan.fit_predict(self.signatures)
        self.corpus_df = pd.DataFrame({"doc_id": self.ids, "cluster": self.clusters})

        self.centroids = {}
        for cluster_label in np.unique(self.clusters):
            if cluster_label == -1:
                continue
            mask = self.clusters == cluster_label
            self.centroids[cluster_label] = np.mean(self.signatures[mask], axis=0)

        self.min_hash_cursor = len(keys)
        print(f"[SketchRetriever] Index ready ({len(keys)} docs, "
              f"{len(self.centroids)} clusters).")

    def update(self, document: dict | int = 1):
        """Append a single document and re-cluster.

        Accepts either a {"key":…,"data":…} dict (streaming interface used by
        the benchmark) or the legacy integer form used by build_corpus().
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
            # legacy: extend from pre-loaded min_hash_docs by count
            n = document
            self.ids.extend(self.min_hash_docs["doc_ids"][self.min_hash_cursor:self.min_hash_cursor + n])
            new_sigs = self.min_hash_docs["signatures"][self.min_hash_cursor:self.min_hash_cursor + n]
            self.signatures = np.vstack([self.signatures, new_sigs])
            self.min_hash_cursor += n

        self.clusters = self.dbscan.fit_predict(self.signatures)
        self.corpus_df = pd.DataFrame({"doc_id": self.ids, "cluster": self.clusters})
        self.centroids = {}
        for cluster_label in np.unique(self.clusters):
            if cluster_label == -1:
                continue
            mask = self.clusters == cluster_label
            self.centroids[cluster_label] = np.mean(self.signatures[mask], axis=0)

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

        if not self.centroids:
            return []

        best_cluster = None
        best_dist = float('inf')
        for cluster_label, centroid_sig in self.centroids.items():
            centroid_int = np.round(centroid_sig).astype(np.uint64)
            dist = np.sum(centroid_int != query_sig)
            if dist < best_dist:
                best_dist = dist
                best_cluster = cluster_label

        doc_ids = self.corpus_df[self.corpus_df['cluster'] == best_cluster]['doc_id'].tolist()
        return doc_ids

    def _set_min_hash_docs(self):
        try:
            self.min_hash_docs = np.load(COLLECTION_MIN_HASH)
        except FileNotFoundError:
            self._generate_min_hash_docs()

    def _generate_min_hash_docs(self):
        keys = []
        sketches = []
        for i, document in enumerate(load_jsonl(COLLECTION_JSONL)):
            print(f'\r{i / 5_000:2f}% done', end='', flush=True)
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
        self.min_hash_docs = np.load(COLLECTION_MIN_HASH)
