from typing import Any, Tuple

from logic.constants import COLLECTION_JSONL, TF_IDF_VECTORIZER, TF_IDF_MATRIX
from logic.processing.file_utils import load_jsonl_data, load_processed_file

from logic.stemming_utils import stemmed_stop_words, StemmingTokenizer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(corpus_size: int) -> Tuple[TfidfVectorizer, Any]:
    print('building vectorizer')
    vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                 tokenizer=StemmingTokenizer(),
                                 stop_words=stemmed_stop_words(),
                                 )

    data_stream = load_jsonl_data(COLLECTION_JSONL, corpus_size)
    tfidf_matrix = vectorizer.fit_transform(data_stream)
    joblib.dump(vectorizer, f'{TF_IDF_VECTORIZER}_{corpus_size}.joblib')
    joblib.dump(tfidf_matrix, f'{TF_IDF_MATRIX}_{corpus_size}.joblib')
    print('vectorizer built')
    return vectorizer, tfidf_matrix


def load_or_build_tfidf(corpus_size: int) -> Tuple[TfidfVectorizer, Any]:
    vec_path = f'{TF_IDF_VECTORIZER}_{corpus_size}.joblib'
    mat_path = f'{TF_IDF_MATRIX}_{corpus_size}.joblib'
    vectorizer = load_processed_file(vec_path)
    matrix = load_processed_file(mat_path)
    if vectorizer is None or matrix is None:
        return build_vectorizer(corpus_size)
    print('Vectorizer already exists. Loading from file')
    return vectorizer, matrix
