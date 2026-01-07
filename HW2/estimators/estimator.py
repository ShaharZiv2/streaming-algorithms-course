from abc import ABC, abstractmethod
import numpy as np


class Estimator(ABC):
    def __init__(self):
        self.__vectorized_update = np.vectorize(self.update)

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def estimate(self):
        pass

    @abstractmethod
    def report(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def vectorized_update(self, vector):
        return self.__vectorized_update(vector)