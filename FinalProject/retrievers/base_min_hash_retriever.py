from abc import abstractmethod

from logic.processing.file_utils import load_processed_file
from retrievers.base_retriever import BaseRetriever


class BaseMinHashRetriever(BaseRetriever):

    def __init__(self, corpus_initial_size: int = 100_000):
        self.docs = None
        self.ids = None
        self.signatures = None
        self.min_hash_cursor = 0
        self.corpus_size = corpus_initial_size
        super().__init__(lazy=(corpus_initial_size == 0))

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

    def _load_minhash_corpus(self):
        self.docs = load_processed_file(self._collection_path) or self._generate_docs()
        self.ids = self.docs['doc_ids'][:self.corpus_size]
        self.signatures = self.docs['signatures'][:self.corpus_size]
        self.min_hash_cursor = self.corpus_size
