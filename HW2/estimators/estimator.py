from abc import ABC, abstractmethod


class Estimator(ABC):

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
