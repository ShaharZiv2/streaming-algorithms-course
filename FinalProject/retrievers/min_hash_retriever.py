from logic.constants import COLLECTION_MIN_HASH, COLLECTION_JSONL, SEED, MIN_HASH_CORPUS_DIR
from logic.processing.file_utils import load_jsonl

from logic.stemming_utils import stemmed_stop_words, StemmingTokenizer
from retrievers.base_min_hash_retriever import BaseMinHashRetriever
import numpy as np
from datasketch import MinHash
from sklearn.feature_extraction.text import CountVectorizer


class MinHashRetriever(BaseMinHashRetriever):

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
        return MIN_HASH_CORPUS_DIR

    def update(self, num_updates: int = 1):
        self.ids.extend(self.docs['doc_ids'][self.min_hash_cursor:num_updates])
        self.signatures.extend(self.docs['signatures'][self.min_hash_cursor:num_updates])
        self.min_hash_cursor += num_updates

        self.clusters = self.dbscan.fit_predict(self.signatures)
        self.corpus_size += num_updates

    def retrieve(self, query):
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
