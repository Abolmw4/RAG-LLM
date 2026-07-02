import unittest
from src.data_ingestion import DataIngestion

class MyTestCase(unittest.TestCase):
    def test_load_pdf_file(self):
        data_igst = DataIngestion(file_path="/home/abolfazl/Documents/rag-llm-project/docs/sodt+++.pdf")
        result = data_igst.load_document()

        print(result[0].metadata)


if __name__ == '__main__':
    unittest.main()
