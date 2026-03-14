from abc import abstractmethod

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from logic.processing.file_utils import load_processed_file
from retrievers.base_retriever import BaseRetriever


class BaseMinHashRetriever(BaseRetriever):

    def __init__(self, dbscan_eps: float = 0.85, corpus_initial_size: int = 100_000):
        self.docs = None
        self.ids = None
        self.signatures = None
        self.clusters = None
        self.centroids = {}
        self.corpus_df = None
        self.min_hash_cursor = 0
        self.corpus_size = corpus_initial_size

        min_samples = max(2, corpus_initial_size // 100_000)
        self.dbscan = DBSCAN(eps=dbscan_eps, min_samples=min_samples, metric='hamming', n_jobs=-1)
        super().__init__()

    @property
    @abstractmethod
    def _collection_path(self) -> str:
        """Path to the .npz file with doc_ids and signatures."""

    @property
    @abstractmethod
    def _corpus_dir(self) -> str:
        """Directory for saving/loading clustered corpus files."""

    @abstractmethod
    def _generate_docs(self):
        """Generate min-hash documents, save them, and return the loaded result."""

    def build_corpus(self):
        self.docs = load_processed_file(self._collection_path) or self._generate_docs()
        self.ids = self.docs['doc_ids'][:self.corpus_size]
        self.signatures = self.docs['signatures'][:self.corpus_size]
        self.min_hash_cursor = self.corpus_size

        self.clusters = (load_processed_file(f'{self._corpus_dir}/{self.corpus_size}_{self.dbscan.eps}.npy')
                         or self._fit_save_dbscan())

        self.corpus_df = pd.DataFrame({'doc_id': self.ids, 'cluster': self.clusters})
        self._compute_centroids()

    def _compute_centroids(self):
        self.centroids = {}
        for cluster_label in np.unique(self.clusters):
            if cluster_label == -1:
                continue
            mask = self.clusters == cluster_label
            self.centroids[cluster_label] = np.mean(self.signatures[mask], axis=0)

    def _fit_save_dbscan(self):
        clusters = self.dbscan.fit_predict(self.signatures)
        np.save(f'{self._corpus_dir}/{self.corpus_size}_{self.dbscan.eps}', arr=clusters)
        return clusters

    def _find_best_cluster(self, query_sig):
        if not self.centroids:
            return None

        best_cluster = None
        best_dist = float('inf')
        for cluster_label, centroid_sig in self.centroids.items():
            centroid_int = np.round(centroid_sig).astype(np.uint64)
            dist = np.sum(centroid_int != query_sig)
            if dist < best_dist:
                best_dist = dist
                best_cluster = cluster_label
        return best_cluster

    def _get_cluster_doc_ids(self, cluster_label):
        if cluster_label is None:
            return []
        return self.corpus_df[self.corpus_df['cluster'] == cluster_label]['doc_id'].tolist()
