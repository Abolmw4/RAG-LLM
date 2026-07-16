import unittest
from unittest.mock import Mock
import numpy as np

from langchain_core.documents import Document

from src.retriver.dense_retriver import DenseRetriver
from src.schemas.retrived_document import RetrivedDocument


class TestDenseRetriever(unittest.TestCase):

    def setUp(self):

        # Mock Embedding
        self.embedding = Mock()

        # Mock Vector Store
        self.vector_store = Mock()

        # Dense Retriever
        self.retriever = DenseRetriver(embedding=self.embedding, vector_store=self.vector_store)

        # Fake embedding
        self.fake_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # Fake retrieved docs
        self.fake_results = [RetrivedDocument(document=Document(page_content="Deep Learning"), score=0.95, doc_id=0), RetrivedDocument(document=Document(page_content="Machine Learning"), score=0.87, doc_id=1)]

        self.embedding.embed_query.return_value = self.fake_embedding
        self.vector_store.search_with_score.return_value = self.fake_results

    def test_retrieve_returns_documents(self):

        result = self.retriever.retrive(qeury="Deep Learning", k=2)

        self.assertEqual(result, self.fake_results)

    def test_embedding_called_once(self):

        self.retriever.retrive(qeury="Deep Learning", k=2)

        self.embedding.embed_query.assert_called_once_with(query="Deep Learning")

    def test_vector_store_called_once(self):

        self.retriever.retrive(
            qeury="Deep Learning",
            k=2
        )

        self.vector_store.search_with_score.assert_called_once_with(
            self.fake_embedding,
            k=2
        )

    def test_output_type(self):

        result = self.retriever.retrive(
            qeury="Deep Learning",
            k=2
        )

        self.assertIsInstance(result, list)

        self.assertIsInstance(
            result[0],
            RetrivedDocument
        )


if __name__ == "__main__":
    unittest.main()