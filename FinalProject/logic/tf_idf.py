from logic.constants import COLLECTION_JSONL, TF_IDF_VECTORIZER, TF_IDF_MATRIX
from logic.processing.file_utils import load_jsonl_data

from logic.stemming_utils import stemmed_stop_words, StemmingTokenizer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(corpus_size: int) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                 tokenizer=StemmingTokenizer(),
                                 stop_words=stemmed_stop_words(),
                                 )

    data_stream = load_jsonl_data(COLLECTION_JSONL, corpus_size)
    tfidf_matrix = vectorizer.fit_transform(data_stream)
    joblib.dump(vectorizer, TF_IDF_VECTORIZER)
    joblib.dump(tfidf_matrix, TF_IDF_MATRIX)
    return vectorizer
