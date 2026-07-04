from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import List


class PromptBuilder(ABC):
    
    @abstractmethod
    def build_prompt(self, question: str, docs: List[Document]) -> str:
        pass
