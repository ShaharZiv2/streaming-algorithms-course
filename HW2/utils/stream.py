from enum import StrEnum
import pandas as pd
import csv

END_TIME_COLUMN = 29

class SourceFiles(StrEnum):
    TRAINING = 'datasets/raw_data/training.csv'
    TESTING = 'datasets/raw_data/testing.csv'

class Stream:

    def __init__(self, data_file_path: SourceFiles, window_size: int):
        self.file = open(data_file_path, 'r')
        self.csv_reader = csv.reader(self.file, delimiter=',')
        self.headers = self.csv_reader.__next__()
        self.window_size = window_size

    def __iter__(self):
        return self

    def __next__(self):
        window = []

        try:
            first_line = line = self.csv_reader.__next__()
            window.append(first_line)

            starting_timestamp = int(first_line[END_TIME_COLUMN])
            while int(line[END_TIME_COLUMN]) - starting_timestamp < self.window_size:
                line = self.csv_reader.__next__()
                window.append(line)
        except StopIteration:
            if window:
                return pd.DataFrame(window, columns=self.headers).astype(dtype='string')
            raise

        return pd.DataFrame(window, columns=self.headers).astype(dtype='string')