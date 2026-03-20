from retrievers.prob_min_hash_lsh_retriever import ProbMinHashLshRetriever


def main():
    min_hash_lsh_retriever = ProbMinHashLshRetriever()
    res = min_hash_lsh_retriever.retrieve("what changes are seen in water balance as we get older\n")
    pass

if __name__ == '__main__':
    main()
