from abc import ABC


class Estimator(ABC):
    def __init__(self):
        pass
    def update(self):
        pass
    def estimate(self):
        pass
    def report(self):
        pass