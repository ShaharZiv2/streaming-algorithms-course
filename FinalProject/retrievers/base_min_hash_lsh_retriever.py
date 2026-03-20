from abc import ABC

import numpy as np
import pickle as pkl
from datasketch import MinHashLSH, MinHash

from logic.processing.file_utils import load_processed_file
from retrievers.base_min_hash_retriever import BaseMinHashRetriever


class BaseMinHashLshRetriever(BaseMinHashRetriever, ABC):

    def __init__(self, min_hash_lsh_eps: float = 0.85, corpus_initial_size: int = 100_000):
        self.index = None

        self.min_hash_lsh_eps = min_hash_lsh_eps
        super().__init__(corpus_initial_size=corpus_initial_size)

    def build_corpus(self):
        self._load_minhash_corpus()
        self.index = load_processed_file(f'{self._corpus_dir}/{self.corpus_size}_{self.min_hash_lsh_eps}.pkl')
        if self.index is None:
            self._fit_save_min_hash_lsh()

    def _fit_save_min_hash_lsh(self):
        self.index = MinHashLSH(threshold=1 - self.min_hash_lsh_eps)
        with self.index.insertion_session() as session:
            for doc_id, signature in zip(self.ids, self.signatures):
                session.insert(doc_id, MinHash(num_perm=128, hashvalues=signature))

        with open(f'{self._corpus_dir}/{self.corpus_size}_{self.min_hash_lsh_eps}.pkl', 'wb') as file:
            pkl.dump(self.index, file)
