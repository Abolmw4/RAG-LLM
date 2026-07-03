from src.data_ingestion import DataIngestion
from src.chunker import Chunker
import unittest


DATA_DIR = "/home/abolfazl/Documents/rag-llm-project/docs"

class MyTestCase(unittest.TestCase):

    def test_something(self):
        documents = DataIngestion(DATA_DIR).load_documents()
        chunker = Chunker(chunk_size=1000, chunk_overlap=150)
        
        chunks = chunker.split_documents(documents)

        print("documents:", len(documents))
        print("chunks:", len(chunks))

        print("\ntenth chunk content:")
        print(chunks[10].page_content[:500])

        print("\ntheth chunk metadata:")
        print(chunks[10].metadata)

if __name__ == '__main__':
    unittest.main()
