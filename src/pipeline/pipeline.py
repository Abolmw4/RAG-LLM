from abc import ABC, abstractmethod

class PipeLine(ABC):
    @abstractmethod
    def ask(self, question: str) -> str:
        pass