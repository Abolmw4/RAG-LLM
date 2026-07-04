import unittest
from unittest.mock import Mock, MagicMock
from src.retriver.retriver import SimpleRetriver
from langchain_core.documents import Document
import numpy as np


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.embdder = MagicMock()
        self.vector_store = MagicMock()
        
        self.retriver = SimpleRetriver(self.embdder, self.vector_store)
        
    def test_retrieve_returns_documents(self):

        docs = [Document(page_content="شبکه های cnn"), Document(page_content="تشخیص شی")]

        self.embdder.embed_query.return_value = np.random.rand(1024)

        self.vector_store.search.return_value = docs

        result = self.retriver.retrive("شبکه های cnn")
        self.assertEqual(result, docs)
    
    def test_embed_query_called(self):

        self.embdder.embed_query.return_value = np.random.rand(1024)

        self.vector_store.search.return_value = []

        self.retriver.retrive("hello")

        self.embdder.embed_query.assert_called_once_with(query="hello")
        
    def test_search_called(self):

        vector = np.random.rand(1024)

        self.embdder.embed_query.return_value = vector

        self.vector_store.search.return_value = []

        self.retriver.retrive("hello")

        self.vector_store.search.assert_called_once()
    
    def test_top_k(self):

        vector = np.random.rand(1024)

        self.embdder.embed_query.return_value = vector

        self.vector_store.search.return_value = []

        self.retriver.retrive("hello", k=3)

        self.vector_store.search.assert_called_once_with(vector, k=3)
