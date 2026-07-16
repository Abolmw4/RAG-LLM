from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from src.schemas.retrived_document import RetrivedDocument

class Retriver(ABC):
    @abstractmethod
    def retrive(self, qeury: str, k: int) -> List[RetrivedDocument]:
        pass
