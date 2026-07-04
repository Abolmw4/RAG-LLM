from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class Retriver(ABC):
    
    @abstractmethod
    def retiver(self, qeury: str, k: int) -> List[Document]:
        pass