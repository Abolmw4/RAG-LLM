from typing import List
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

import numpy as np
import torch

class Embedding:
    def __init__(self, model_name: str, device: str | None) -> None:
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)
        
    @property
    def device(self) -> str:
        return self._device
    
    @device.setter
    def device(self, value: str) -> None:
        if value is None:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        elif value.lower() in ["cuda", "cpu"]:
            self._device = value if value == 'cuda' and torch.cuda.is_available() else 'cpu'

        else:
            raise ValueError(f"'device' must be 'cuda' or 'cpu'")
    
    def embed_documents(self, chunks: List[Document], batch_size: int=32, normalize_embeddings: bool=True, show_progress_bar: bool=True, convert_to_numpy: bool=True) -> np.ndarray:
        if not chunks:
            raise ValueError("chunks cannot be empty.")
        
        texts: List[str] = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(texts, batch_size=batch_size,
                                       normalize_embeddings=normalize_embeddings,
                                       show_progress_bar=show_progress_bar,
                                       convert_to_numpy=convert_to_numpy)
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        
        embedding = self.model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        return embedding