from abc import ABC, abstractmethod
from typing import Dict

class SketchManagerBase(ABC):

    @abstractmethod
    def sketch(self, event: Dict[str, str]) -> None:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def estimate(self):
        pass

    @abstractmethod
    def calculate_total_memory(self) -> int:
        pass