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

    def build_corpus_from_docs(self, docs: list[dict]):
        """Build the MinHash LSH index from a pre-loaded list of {"key":…,"data":…} dicts."""
        from datasketch import MinHashLSH
        print(f"[MinHashLshRetriever] Building MinHash signatures for {len(docs)} documents…")
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
        self.min_hash_cursor = len(keys)

        print(f"[MinHashLshRetriever] Building LSH index (threshold={1 - self.min_hash_lsh_eps:.2f})…")
        self.index = MinHashLSH(threshold=1 - self.min_hash_lsh_eps, num_perm=128)
        with self.index.insertion_session() as session:
            for doc_id, sig in zip(self.ids, self.signatures):
                session.insert(doc_id, MinHash(num_perm=128, hashvalues=sig))
        print(f"[MinHashLshRetriever] LSH index ready ({len(keys)} docs).")

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
        """Append a single document to the LSH index.

        Accepts either a {"key":…,"data":…} dict (benchmark streaming interface)
        or the legacy integer form (no-op for LSH).
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
            doc_id = document["key"]
            self.ids.append(doc_id)
            new_sig = m.hashvalues.reshape(1, -1)
            self.signatures = np.vstack([self.signatures, new_sig])
            if self.index is not None:
                try:
                    self.index.insert(doc_id, m)
                except Exception:
                    pass
        # legacy integer path is a no-op for LSH
