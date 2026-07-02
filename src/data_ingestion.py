from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from typing import List


class DataIngestion:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
    
    def _load_pdf(self) -> List[Document]:
        print("info:", f"Loading PDF: {self.file_path.split('/')[-1]}")

        loader = PyPDFLoader(str(self.file_path))

        documents = loader.load()

        print("info:", f"Loaded {len(documents)} pages.")

        return documents
    
    def load_document(self) -> List[Document]:
        suffix = self.file_path.split('/')[-1].split('.')[-1]
        match suffix:
            case "pdf":
                return self._load_pdf()
            case _:
                print("Unknow file")        
        