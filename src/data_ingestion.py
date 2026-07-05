from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from typing import List
from src.utils.utils import logger


class DataIngestion:
    SUPPORTED_FILES = ['.pdf']
    def __init__(self, path: str) -> None:
        self.path = Path(path)
    
    def _load_pdf(self, pdf_path: Path) -> List[Document]:
        logger.info(f"{__class__.__name__}: Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))

        documents = loader.load()

        for doc in documents:
            doc.metadata["source"] = pdf_path.name
            
        logger.info(f"{__class__.__name__}: Loaded {len(documents)} pages.")
        return documents
    
    def load_documents(self) -> List[Document]:
        documents = []

        if self.path.is_file():
            if self.path.suffix.lower() not in self.SUPPORTED_FILES:
                logger.error(f"{__class__.__name__}: Unsupported file: {self.path.suffix}")
                raise ValueError(f"Unsupported file: {self.path.suffix}")
            return self._load_pdf(self.path)

        elif self.path.is_dir():
            pdf_files = sorted(self.path.glob("*.pdf"))
            if not pdf_files:
                logger.error(f"{__class__.__name__}: No PDF files found.")
                raise ValueError("No PDF files found.")
            for pdf in pdf_files:
                documents.extend(self._load_pdf(pdf))
            return documents
        else:
            logger.error(f"{__class__.__name__}: Unexpected path.")
            raise RuntimeError("Unexpected path.")