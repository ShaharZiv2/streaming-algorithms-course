from retrievers.classic_retriever import ClassicRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.min_hash_dbscan_retriever import MinHashDbscanRetriever
from retrievers.prob_min_hash_dbscan_retriever import ProbMinHashDbscanRetriever


def main():
    sketch_retriever = ProbMinHashDbscanRetriever()
    sketch_retriever = MinHashDbscanRetriever()
    classic_retriever = ClassicRetriever(top_k=10)
    bm25_retriever = BM25Retriever(top_k=10)

if __name__ == '__main__':
    main()



# base line retriever = bm25
# lsh minhash retriever =
# metrics: time taken to build corpus, time taken to retrieve, memory, calculate prescion in comparison to classic and bm25 base line to get prescion
# from that we will get the bias and variance of the sketch retriever

