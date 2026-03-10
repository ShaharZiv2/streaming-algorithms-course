from abc import ABC, abstractmethod

from logic.constants import COLLECTION_JSONL


class BaseRetriever(ABC):

    def __init__(self, lazy: bool = False):
        if not lazy:
            self.corpus = self.build_corpus()

    @abstractmethod
    def build_corpus(self):
        """Builds the initial corpus for retrieval upon a query"""

    @abstractmethod
    def update(self, document):
        """Adds another document to the corpus"""

    @abstractmethod
    def retrieve(self, query):
        """Retrieves the relevant documents for the corpus"""