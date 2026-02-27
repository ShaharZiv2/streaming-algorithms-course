TSV_FILES_DIR = 'datasets/tsv'
JSONL_FILES_DIR = 'datasets/jsonl'
PROCESSED_FILES_DIR = 'datasets/processed'
PROCESSED_MIN_HASH_FILES_DIR = f'{PROCESSED_FILES_DIR}/min_hash'

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


SEED = 42
