from src.vector_store.vector_store_base import BaseVectorStore
from typing import List
from langchain_core.documents import Document
import faiss
import numpy as np
import pickle
from src.utils.utils import logger
from src.schemas.retrived_document import RetrivedDocument
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
            logger.error(f"{__class__.__name__}: 'dimention' has a problem in type and value")
            raise ValueError(f"'dimention' must be grater than 0 and must be int")
    
    def add(self, embeddings: np.ndarray, docs: List[Document]) -> None:
        if embeddings.shape[0] != len(docs):
            logger.error(f"{__class__.__name__}: number of embeddings and Documents not match.")
            raise ValueError("number of embeddings and Documents must be match.")

        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.documents.extend(docs)
        
    @classmethod
    def load(cls, index_path: str, document_path: str):
        try:
            index = faiss.read_index(index_path)
            with open(document_path, 'rb') as file:
                documents = pickle.load(file)
            store = cls(index.d)
            store.index = index
            store.documents = documents
            logger.debug(f"{__class__.__name__}: load index and document sucessfully")
            return store
        except RuntimeError as error:
            logger.error(f"{__class__.__name__}: when loading index and document error '{error}' occured")
    
    def save(self, index_path: str, document_path: str) -> None:
        try:
            faiss.write_index(self.index, index_path)
            with open(document_path, 'wb') as file:
                pickle.dump(self.documents, file)
            logger.debug(f"{__class__.__name__}: index and document saved sucessfully")
        except RuntimeError as error:
            logger.error(f"{__class__.__name__}: save index and document error '{error}' occured")
    
    def search(self, query: np.ndarray, k: int=5) -> List[Document]:
        if self.index.ntotal == 0:
            logger.error(f"{__class__.__name__}: Vector store is empty")
            raise ValueError("Vector store is empty")
        
        result: List[Document] = []
        if query.ndim == 1:
            query_embedding = np.expand_dims(query, axis=0)
        
        _, indices = self.index.search(query_embedding.astype(np.float32), k=k)

        for indx in indices[0]:
            if indx == -1:
                continue
            result.append(self.documents[indx])
        return result
    
    def search_with_score(self, query: np.ndarray, k: int=5) -> List[RetrivedDocument]:
        if self.index.ntotal == 0:
            logger.error(f"{__class__.__name__}: Vector store is empty")
            raise ValueError("Vector store is empty")
        
        result: List[RetrivedDocument] = []
        if query.ndim == 1:
            query_embedding = np.expand_dims(query, axis=0)
        
        scores, indices = self.index.search(query_embedding.astype(np.float32), k=k)

        for indx, score in zip(indices[0], scores[0]):
            if indx == -1:
                continue
            result.append(RetrivedDocument(document=self.documents[indx], score=float(score), doc_id=indx))
        return result
