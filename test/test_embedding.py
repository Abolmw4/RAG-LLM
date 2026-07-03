import unittest
from src.embeddings import Embedding
from langchain_core.documents import Document
from typing import List

path_file = "/home/abolfazl/Documents/rag-llm-project/docs"
class MyTestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.embedder = Embedding(model_name="BAAI/bge-m3", device="cuda")
    
    def test_model_loaded(self):
        self.assertIsNotNone(self.embedder.model)
    
    
    def test_device(self):
        self.assertEqual(self.embedder.device, "cuda")
        
    def test_invalid_device(self):
        with self.assertRaises(ValueError):
            Embedding(model_name="BAAI/bge-m3", device="abc")
    
    def test_embed_one_document(self):

        docs = [Document(page_content="Artificial Intelligence")]

        embeddings = self.embedder.embed_documents(docs)

        self.assertEqual(embeddings.shape, (1, 1024))
    
    def test_embed_multiple_documents(self):

        docs = [Document(page_content="Artificial Intelligence"), Document(page_content="Deep Learning"), Document(page_content="Computer Vision")]

        embeddings = self.embedder.embed_documents(docs)

        self.assertEqual(embeddings.shape, (3, 1024))
