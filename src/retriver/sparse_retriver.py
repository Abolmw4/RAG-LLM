from src.schemas.retrived_document import RetrivedDocument
from src.retriver.retriver_base import Retriver
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List


class SparseRetriver(Retriver):
    def __init__(self, documents: List[Document]):
        super().__init__()
        self.documents = documents
        self.tokenized_documets = [self._tokenizer(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_documets)
    
    
    def _tokenizer(self, text: str) -> List[str]:
        return text.lower().split()
    
    
    def retrive(self, qeury: str, k: int) -> List[RetrivedDocument]:
        
        qeury_tokenized = self._tokenizer(qeury)
        socres = self.bm25.get_scores(qeury_tokenized)
        k = min(k, len(self.documents))
        top_indeces = np.argpartition(socres, -k)[-k:]
        
        top_indeces = top_indeces[np.argsort(socres[top_indeces])[::-1]]
        
        results = []
        for idx in top_indeces:
            results.append(RetrivedDocument(document=self.documents[idx], score=float(socres[idx]), doc_id=idx))
        return results
