import unittest
from src.retriver.retriver_base import Retriver
from src.retriver.sparse_retriver import SparseRetriver
from langchain_core.documents import Document
from src.schemas.retrived_document import RetrivedDocument
from typing import List


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.documets: List[Document] = [Document(page_content="Hi I'm abolfazl asghari"),
                                         Document(page_content="I'm best in LLM ans CV"),
                                         Document(page_content="My favorate languages is python and cpp")]
        self.retriver: Retriver = SparseRetriver(documents=self.documets)
    
    def test_return_top_k_documets(self):
        result = self.retriver.retrive(qeury="python langauage", k=2)
        self.assertEqual(len(result), 2)
        
    def test_return_retrived_document(self):
        result = self.retriver.retrive(qeury="python", k=1)
        self.assertIsInstance(result[0], RetrivedDocument)
