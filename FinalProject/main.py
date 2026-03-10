from retrievers.min_hash_retriever import MinHashRetriever
from retrievers.prob_min_hash_retriever import ProbMinHashRetriever
from retrievers.classic_retriever import ClassicRetriever
from retrievers.knn_retriever import KnnRetriever


def main():
    sketch_retriever = ProbMinHashRetriever()
    sketch_retriever = MinHashRetriever()
    classic_retriever = ClassicRetriever(top_k=10)
    knn_retriever = KnnRetriever(top_k=10)

if __name__ == '__main__':
    main()
