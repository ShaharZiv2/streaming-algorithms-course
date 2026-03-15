from retrievers.min_hash_retriever import MinHashRetriever
from retrievers.prob_min_hash_retriever import ProbMinHashRetriever
from retrievers.classic_retriever import ClassicRetriever
from retrievers.bm25_retriever import BM25Retriever


def main():
    sketch_retriever = ProbMinHashRetriever()
    sketch_retriever = MinHashRetriever()
    classic_retriever = ClassicRetriever(top_k=10)
    bm25_retriever = BM25Retriever(top_k=10)

if __name__ == '__main__':
    main()



# base line retriever = bm25
# lsh minhash retriever =
# metrics: time taken to build corpus, time taken to retrieve, recall@k, precision@k, F1@k, MRR@k

