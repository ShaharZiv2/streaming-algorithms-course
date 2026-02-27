from enum import StrEnum
import csv


class SourceFiles(StrEnum):
    TRAINING = 'datasets/raw_data/training.csv'
    TESTING = 'datasets/raw_data/testing.csv'

    TRAINING_LIGHT = 'datasets/light/training.csv'
    TESTING_LIGHT = 'datasets/light/testing.csv'


class Stream:

    def __init__(self, data_file_path: SourceFiles, window_size: int):
        self.file = open(data_file_path, 'r')
        self.csv_reader = csv.DictReader(self.file, delimiter=',')
        self.window_size = window_size
        self.event = next(self.csv_reader)
        self.window_starting_time = int(self.event['Stime'])
        self.window_done: bool = False

    def __iter__(self):
        return self

    def __next__(self):
        event = self.event

        self.event = next(self.csv_reader)
        self.window_done = (int(self.event['Stime']) - self.window_starting_time) > self.window_size

        return event

    def reset_window(self) -> None:
        self.window_starting_time = int(self.event['Stime'])
        self.window_done = False
