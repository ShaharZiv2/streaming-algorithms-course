from enum import StrEnum
import pandas as pd


class SourceFiles(StrEnum):
    TRAINING = 'datasets/UNSW_NB15_training-set.csv'
    TESTING = 'datasets/UNSW_NB15_testing-set.csv'

class Stream:

    def __init__(self, data_file_path: SourceFiles, window_size: int):
        # Placeholder until windowing is clear
        self.file_reader = pd.read_csv(data_file_path, chunksize=window_size)
        self.window_size = window_size
        self.window = None

    def __iter__(self):
        return self

    def __next__(self):
        # When windowing will be more clear, fix this
        return self.file_reader.get_chunk()