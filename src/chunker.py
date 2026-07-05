from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.utils import logger
from typing import List

class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if self.chunk_overlap >= self.chunk_size:
            logger.error(f"{__class__.__name__}: 'chunk_overlap' greater than 'chunk_size'")
            raise ValueError(f"'chunk_overlap' must be less than 'chunk_size'")
    
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separators=["\n\n", "\n", "؟", "!", ".", "،", " ", ""],)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size
    
    @chunk_size.setter
    def chunk_size(self, value: int) -> None:
        if isinstance(value, int):
            if value > 0:
                self._chunk_size = value
            else:
                logger.error(f"{__class__.__name__}: 'chunk_size' less than or eqeal 0")
                raise ValueError(f"'chunk_size' must be grater than 0")
        else:
            logger.error(f"{__class__.__name__}: 'chuk_size' type has a problem")
            raise TypeError(f"'chuk_size' must be int type")
    
    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap
    
    @chunk_overlap.setter
    def chunk_overlap(self, value: int) -> None:
        if isinstance(value, int):
            if value >= 0:
                self._chunk_overlap = value
            else:
                logger.error(f"{__class__.__name__}: 'chunk_overlap' less than 0")
                raise ValueError("'chunk_overlap' must be grater than 0")
        else:
            logger.error(f"{__class__.__name__}: 'chunk_overlap' type has a problem")
            raise TypeError(f"'chunk_overlap' must be int type")
        
    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not isinstance(documents, list):
            logger.error(f"{__class__.__name__}: 'documents' not list of Documets")
            raise TypeError("'documents' must be List of Documet")
        if not documents:
            logger.error(f"{__class__.__name__}: 'documents' empty")
            raise ValueError("'documents' must be not empty")
        
        chunks = self.text_splitter.split_documents(documents)
        
        for indx, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "unknown")
            chunk.metadata["chunk_id"] = f"{source}_page_{page}_chunk_{indx}"
            chunk.metadata["chunk_length"] = len(chunk.page_content)
        return chunks
    