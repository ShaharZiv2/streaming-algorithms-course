TSV_FILES_DIR = 'datasets/tsv'
FULLDOCS_TSV_GZ = f'{TSV_FILES_DIR}/fulldocs.tsv.gz'
JSONL_FILES_DIR = 'datasets/jsonl'
PROCESSED_FILES_DIR = 'datasets/processed'
PROCESSED_MIN_HASH_FILES_DIR = f'{PROCESSED_FILES_DIR}/min_hash'
PROCESSED_PROB_MIN_HASH_FILES_DIR = f'{PROCESSED_FILES_DIR}/prob_min_hash'

COLLECTION_URL = 'https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz'
QUERIES_URL = 'https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz'
QRELS_URL = 'https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv'

COLLECTION_TSV = f'{TSV_FILES_DIR}/collection.tsv'
QUERIES_TSV = f'{TSV_FILES_DIR}/queries.tsv'
QRELS_TSV = f'{TSV_FILES_DIR}/qrels.tsv'

COLLECTION_JSONL = f'{JSONL_FILES_DIR}/collection.jsonl'
QUERIES_JSONL = f'{JSONL_FILES_DIR}/queries.jsonl'
QRELS_JSONL = f'{JSONL_FILES_DIR}/qrels.jsonl'

COLLECTION_MIN_HASH = f'{PROCESSED_MIN_HASH_FILES_DIR}/collection.npz'
QUERIES_MIN_HASH = f'{PROCESSED_MIN_HASH_FILES_DIR}/queries.npz'
QRELS_MIN_HASH = f'{PROCESSED_MIN_HASH_FILES_DIR}/qrels.npz'

COLLECTION_PROB_MIN_HASH = f'{PROCESSED_PROB_MIN_HASH_FILES_DIR}/collection.npz'
QUERIES_PROB_MIN_HASH = f'{PROCESSED_PROB_MIN_HASH_FILES_DIR}/queries.npz'
QRELS_PROB_MIN_HASH = f'{PROCESSED_PROB_MIN_HASH_FILES_DIR}/qrels.npz'

DBSCAN_CORPUS_DIR = f'{PROCESSED_FILES_DIR}/dbscan_clustered_corpus'
DBSCAN_MIN_HASH_CORPUS_DIR = f'{DBSCAN_CORPUS_DIR}/min_hash'
DBSCAN_PROB_MIN_HASH_CORPUS_DIR = f'{DBSCAN_CORPUS_DIR}/prob_min_hash'

LSH_CORPUS_DIR = f'{PROCESSED_FILES_DIR}/lsh_clustered_corpus'
MIN_HASH_LSH_CORPUS_DIR = f'{LSH_CORPUS_DIR}/min_hash'
PROB_MIN_HASH_LSH_CORPUS_DIR = f'{LSH_CORPUS_DIR}/prob_min_hash'

TF_IDF_DIR = f'{PROCESSED_FILES_DIR}/tf_idf'
TF_IDF_VECTORIZER = f'{TF_IDF_DIR}/tf_idf_vectorizer'
TF_IDF_MATRIX = f'{TF_IDF_DIR}/tf_idf_matrix'


SEED = 42
