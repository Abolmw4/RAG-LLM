from src.retriver.retriver_base import Retriver
from src.embeddings import Embedding
from src.vector_store.vector_store_base import BaseVectorStore
from langchain_core.documents import Document
import numpy as np
from typing import List


class SimpleRetriver(Retriver):
    def __init__(self, embedding: Embedding, vectore_store: BaseVectorStore):
        super().__init__()

        self.embedder = embedding
        self.vector_store = vectore_store
    
    def retiver(self, qeury: str, k: int=5) -> List[Document]:
        query_embedding: np.ndarray = self.embedder.embed_query(query=qeury)
        reuslt: List[Document] = self.vector_store.search(query_embedding, k=k)
        return reuslt
