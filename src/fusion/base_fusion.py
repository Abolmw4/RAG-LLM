from abc import ABC, abstractmethod
from src.schemas.retrived_document import RetrivedDocument
from typing import List

class BaseFusion(ABC):
    @abstractmethod
    def merge(self, dense_result: List[RetrivedDocument], spars_result: List[RetrivedDocument], top_k: int) -> List[RetrivedDocument]:
        pass
