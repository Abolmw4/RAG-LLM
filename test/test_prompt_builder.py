import unittest
from src.prompt_builder.simple_prompt import SimplePrompt
from langchain_core.documents import Document


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.builder = SimplePrompt()
    
    
    def test_build_prompt_return_type(self):

        docs = [Document(page_content="cnn یک نوع شبکه عصبی عمیق است.")]

        prompt = self.builder.build_prompt(question="شبکه cnn چیه؟؟", docs=docs)
        self.assertIsInstance(prompt, str)
