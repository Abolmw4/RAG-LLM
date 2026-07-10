from src.retriver.retriver_base import Retriver
from src.embeddings import Embedding
from src.vector_store.vector_store_base import BaseVectorStore
from src.schemas.retrived_document import RetrivedDocument
from langchain_core.documents import Document
import numpy as np
from typing import List


class DenseRetriver(Retriver):
    def __init__(self, embedding: Embedding, vector_store: BaseVectorStore):
        super().__init__()
        self.embedder = embedding
        self.vector_store = vector_store 
    
    def retrive(self, qeury: str, k: int=5) -> List[RetrivedDocument]:
        query_embedding: np.ndarray = self.embedder.embed_query(query=qeury)
        # reuslt: List[Document] = self.vector_store.search(query_embedding, k=k)
        reuslt: List[RetrivedDocument] = self.vector_store.search_with_score(query_embedding, k=k)
        return reuslt
