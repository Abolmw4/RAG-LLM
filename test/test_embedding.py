import unittest
from src.embeddings import Embedding
from src.data_ingestion import DataIngestion
from src.chunker import Chunker
from langchain_core.documents import Document
from typing import List

path_file = "/home/abolfazl/Documents/rag-llm-project/docs"
class MyTestCase(unittest.TestCase):
    def test_something(self):
        data_instg = DataIngestion(path=path_file)
        chunker = Chunker(chunk_size=500, chunk_overlap=50)
        embedder = Embedding(model_name="BAAI/bge-m3", device="cuda")
        
        docs: List[Document] = data_instg.load_documents()
        chunks: List[Document] = chunker.split_documents(docs)
        result = embedder.embed_documentes(chunks=chunks)
        print(result[0,:].min())
        print(result[0,:].max())
