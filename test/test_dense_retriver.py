import unittest
from src.retriver.retriver_base import Retriver
from src.retriver.dense_retriver import DenseRetriver
from langchain_core.documents import Document
from src.schemas.retrived_document import RetrivedDocument
from typing import List


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.documets: List[Document] = [Document(page_content="Hi I'm abolfazl asghari"),
                                         Document(page_content="I'm best in LLM ans CV"),
                                         Document(page_content="My favorate languages is python and cpp")]
        self.retriver: Retriver = DenseRetriver(EM)
        