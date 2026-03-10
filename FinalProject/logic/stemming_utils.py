import re
from typing import List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import SnowballStemmer

def stemmed_stop_words() -> List[str]:
    stemmer = SnowballStemmer('english')
    stemmed_stop_words_list = [stemmer.stem(word) for word in ENGLISH_STOP_WORDS]
    return stemmed_stop_words_list


class StemmingTokenizer:
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
        self.pattern = re.compile(r'(?u)\b\d+\.\d+\b|\b\w+\b')

    def __call__(self, doc):
        raw_tokens = self.pattern.findall(doc)
        return [self.stemmer.stem(t) for t in raw_tokens]
