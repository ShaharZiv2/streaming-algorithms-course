from abc import ABC

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from logic.processing.file_utils import load_processed_file
from retrievers.base_min_hash_retriever import BaseMinHashRetriever


class BaseMinHashDbscanRetriever(BaseMinHashRetriever, ABC):

    def __init__(self, dbscan_eps: float = 0.85, corpus_initial_size: int = 100_000):
        min_samples = max(2, corpus_initial_size // 100_000)
        self.dbscan = DBSCAN(eps=dbscan_eps, min_samples=min_samples, metric='hamming', n_jobs=-1)
        self.clusters = None
        self.centroids = {}
        self.corpus_df = None
        super().__init__(corpus_initial_size=corpus_initial_size)

    def build_corpus(self):
        self._load_minhash_corpus()

        self.clusters = load_processed_file(f'{self._corpus_dir}/{self.corpus_size}_{self.dbscan.eps}.npy')
        if self.clusters is None:
            self.clusters = self._fit_save_dbscan()

        self.corpus_df = pd.DataFrame({'doc_id': self.ids, 'cluster': self.clusters})
        self._compute_centroids()

    def _compute_centroids(self):
        self.centroids = {}
        for cluster_label in np.unique(self.clusters):
            if cluster_label == -1:
                continue
            mask = self.clusters == cluster_label
            self.centroids[cluster_label] = np.mean(self.signatures[mask], axis=0)

        # Precompute stacked centroid matrix for fast vectorised lookup
        if self.centroids:
            self._centroid_labels = np.array(list(self.centroids.keys()), dtype=np.int64)
            self._centroid_matrix = np.round(
                np.array(list(self.centroids.values()))
            ).astype(np.uint64)   # shape: (n_clusters, num_perm)
        else:
            self._centroid_labels = np.array([], dtype=np.int64)
            self._centroid_matrix = None

    def _fit_save_dbscan(self):
        clusters = self._predict_clusters(self.signatures)
        np.save(f'{self._corpus_dir}/{self.corpus_size}_{self.dbscan.eps}', arr=clusters)
        return clusters

    def _predict_clusters(self, signatures):
        return self.dbscan.fit_predict(signatures)

    def _find_best_cluster(self, query_sig):
        if not self.centroids or self._centroid_matrix is None:
            return None

        # Vectorised Hamming distance: (n_clusters, perm) != (perm,) → rowwise sum
        dists = np.sum(self._centroid_matrix != query_sig, axis=1)  # (n_clusters,)
        best_idx = int(np.argmin(dists))
        return int(self._centroid_labels[best_idx])

    def _get_cluster_doc_ids(self, cluster_label):
        if cluster_label is None:
            return []
        return self.corpus_df[self.corpus_df['cluster'] == cluster_label]['doc_id'].tolist()
