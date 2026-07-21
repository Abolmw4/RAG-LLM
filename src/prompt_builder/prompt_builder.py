from abc import ABC, abstractmethod
from src.schemas.retrived_document import RetrivedDocument
from typing import List


class PromptBuilder(ABC):
    
    @abstractmethod
    def build_prompt(self, question: str, docs: List[RetrivedDocument]) -> str:
        pass
