from logic.constants import COLLECTION_MIN_HASH, COLLECTION_JSONL, SEED, MIN_HASH_LSH_CORPUS_DIR
from logic.processing.file_utils import load_jsonl

from logic.stemming_utils import stemmed_stop_words, StemmingTokenizer
import numpy as np
from datasketch import MinHash
from sklearn.feature_extraction.text import CountVectorizer

from retrievers.base_min_hash_lsh_retriever import BaseMinHashLshRetriever


class MinHashLshRetriever(BaseMinHashLshRetriever):

    def __init__(self, min_hash_lsh_eps: float = 0.85, corpus_initial_size: int = 100_000):
        self.vectorizer = CountVectorizer(ngram_range=(1, 2),
                                          tokenizer=StemmingTokenizer(),
                                          stop_words=stemmed_stop_words())
        super().__init__(min_hash_lsh_eps, corpus_initial_size)

    @property
    def _collection_path(self) -> str:
        return COLLECTION_MIN_HASH

    @property
    def _corpus_dir(self) -> str:
        return MIN_HASH_LSH_CORPUS_DIR

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

        return self.index.query(m)

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
