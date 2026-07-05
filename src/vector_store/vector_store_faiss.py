from src.vector_store.vector_store_base import BaseVectorStore
from typing import List
from langchain_core.documents import Document
import faiss
import numpy as np
import pickle

class FaissVectorStore(BaseVectorStore):
    def __init__(self, dimention: int) -> None:
        self.dimention = dimention
        self.index = faiss.IndexFlatIP(self.dimention)

        self.documents: List[Document] = []
        
    @property
    def dimention(self) -> int:
        return self._dimention
    
    @dimention.setter
    def dimention(self, value: int) -> None:
        if isinstance(value, int) and value > 0:
            self._dimention = value
        else:
            raise ValueError(f"'dimention' must be grater than 0 and must be int")
    
    def add(self, embeddings: np.ndarray, docs: List[Document]) -> None:
        if embeddings.shape[0] != len(docs):
            raise ValueError("number of embeddings and Documents must be match.")

        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.documents.extend(docs)
        
    @classmethod
    def load(cls, index_path: str, document_path: str):
        index = faiss.read_index(index_path)
        with open(document_path, 'rb') as file:
            documents = pickle.load(file)
        
        store = cls(index.d)
        store.index = index
        store.documents = documents
        return store
    
    def save(self, index_path: str, document_path: str) -> None:
        faiss.write_index(self.index, index_path)
        with open(document_path, 'wb') as file:
            pickle.dump(self.documents, file)

    def search(self, query: np.ndarray, k: int=5) -> List[Document]:
        if self.index.ntotal == 0:
            raise ValueError("Vector store is empty")
        
        result: List[Document] = []
        if query.ndim == 1:
            query_embedding = np.expand_dims(query, axis=0)
        
        scores, indices = self.index.search(query_embedding.astype(np.float32), k=k)

        for indx in indices[0]:
            if indx == -1:
                continue
            result.append(self.documents[indx])
        return result