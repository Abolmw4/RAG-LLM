from dataclasses import dataclass
from langchain_core.documents import Document

@dataclass
class RetrivedDocument:
    document: Document
    score: float
