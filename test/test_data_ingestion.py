import unittest
from src.data_ingestion import DataIngestion

DATA_DIR = "/home/abolfazl/Documents/rag-llm-project/docs"

class MyTestCase(unittest.TestCase):
    data_igst = DataIngestion(DATA_DIR)
    result = data_igst.load_documents()
    
    def test_documents_are_loaded(self):
        cls = type(self)
        result = cls.data_igst.load_documents()
        self.assertGreater(len(result), 0)

    def test_document_type(self):
        cls = type(self)
        from langchain_core.documents import Document
        self.assertIsInstance(cls.result[0], Document)

    def test_page_content_is_not_empty(self):
        cls = type(self)
        self.assertTrue(len(cls.result[0].page_content.strip()) > 0)

    def test_metadata_contains_required_fields(self):
        cls = type(self)
        metadata = cls.result[0].metadata

        self.assertIn("source", metadata)
        self.assertIn("page", metadata)
        self.assertIn("page_label", metadata)
        self.assertIn("total_pages", metadata)

    def test_source_is_pdf(self):
        cls = type(self)
        source = cls.result[0].metadata["source"]
        self.assertTrue(source.endswith(".pdf"))
        
if __name__ == '__main__':
    unittest.main()
