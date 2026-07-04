import unittest
from unittest.mock import Mock, MagicMock, create_autospec
from src.pipeline.simple_rag_pipeline import SimpleRAGPipeLine
from langchain_core.documents import Document
from src.retriver.retriver import Retriver
from src.prompt_builder.prompt_builder import PromptBuilder
from src.generators.generator import Generator

class MyTestCase(unittest.TestCase):
    
    def setUp(self):

        self.retriever = create_autospec(Retriver)
        self.prompt_builder = create_autospec(PromptBuilder)
        self.generator = create_autospec(Generator)

        self.pipeline = SimpleRAGPipeLine(
            retriver=self.retriever,
            prompt_builder=self.prompt_builder,
            generator=self.generator
        )
     
    def test_ask_returns_answer(self):

        docs = [Document(page_content="CNN")]

        self.retriever.retrive.return_value = docs

        self.prompt_builder.build_prompt.return_value = "PROMPT"

        self.generator.generate.return_value = "CNN is ..."

        result = self.pipeline.ask("What is CNN?")

        self.assertEqual(result, "CNN is ...")
