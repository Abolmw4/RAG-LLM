import unittest
from langchain_core.documents import Document
from src.vector_store.vector_store_faiss import FaissVectorStore
import numpy as np

class MyTestCase(unittest.TestCase):
    
    def setUp(self):
        self.dimension = 1024
        self.store = FaissVectorStore(self.dimension)
        self.documents = [
            Document(page_content="Abolfal Asghari"),
            Document(page_content="Deep Learning"),
            Document(page_content="Computer Vision"),
        ]

        embeddings = np.random.rand(3, self.dimension).astype(np.float32)

        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.embeddings = embeddings
    

    def test_create_vector_store(self):

        self.assertEqual(self.store.dimention, self.dimension)
    
    
    def test_add_documents(self):
        self.store.add(self.embeddings, self.documents)
        self.assertEqual(self.store.index.ntotal, 3)
        
    
    
    def test_search(self):

        self.store.add(self.embeddings, self.documents)

        result = self.store.search(self.embeddings[0], k=1)
        
        self.assertEqual(len(result), 1)

        self.assertEqual(result[0].page_content, self.documents[0].page_content)
