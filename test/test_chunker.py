from src.data_ingestion import DataIngestion
from src.chunker import Chunker
from langchain_core.documents import Document
from typing import List
import unittest


DATA_DIR = "/home/abolfazl/Documents/rag-llm-project/docs"

class MyTestCase(unittest.TestCase):

    def test_split_documenet(self):
        docs: List[Document] = [Document(page_content="Hi every one I'm Abolfazl Asghari", metdata={'source':"doc1.pdf", 'page':0})]
        chunker = Chunker(chunk_size=1000, chunk_overlap=100)    
        chunks: list[Document] = chunker.split_documents(docs)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, "Hi every one I'm Abolfazl Asghari")
    def test_split_long_document(self):
        docs: List[Document] = [Document(page_content="Hi every one I'm Abolfazl Asghari " * 1000, metdata={'source':"doc1.pdf", 'page':0})]
        chunker = Chunker(chunk_size=500, chunk_overlap=50)    
        chunks: list[Document] = chunker.split_documents(docs)
        self.assertGreater(len(chunks), 1)
        
    def test_metadata_is_preserved(self):
        docs: List[Document] = [Document(page_content="Hello " * 300,metadata={"source": "myfile.pdf", "page": 10, "author": "Abolfazl"})]

        chunker = Chunker(chunk_size=300, chunk_overlap=50)

        chunks = chunker.split_documents(docs)

        self.assertEqual(chunks[0].metadata["source"], "myfile.pdf")
        self.assertEqual(chunks[0].metadata["page"], 10)
        self.assertEqual(chunks[0].metadata["author"], "Abolfazl")
        
    def test_chunk_id_created(self):
        docs: List[Document] = [Document(page_content="Hello " * 300, metadata={"source": "test.pdf", "page": 2})]

        chunker = Chunker(chunk_size=300, chunk_overlap=50)

        chunks = chunker.split_documents(docs)

        for chunk in chunks:
            self.assertIn("chunk_id", chunk.metadata)

if __name__ == '__main__':
    unittest.main()
