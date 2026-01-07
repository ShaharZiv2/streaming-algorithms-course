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
        self.line = self.csv_reader.__next__()
        self.window_size = window_size
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration

        window = []
        starting_timestamp = int(self.line[END_TIME_COLUMN])
        while int(self.line[END_TIME_COLUMN]) - starting_timestamp < self.window_size:
            window.append(self.line)
            try:
                self.line = self.csv_reader.__next__()
            except StopIteration:
                self.finished = True
                if window:
                    return pd.DataFrame(window, columns=self.headers)
                raise


        return pd.DataFrame(window, columns=self.headers)
