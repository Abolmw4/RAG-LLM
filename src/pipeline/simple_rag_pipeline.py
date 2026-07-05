from src.pipeline.pipeline import PipeLine
from src.retriver.retriver_base import Retriver 
from src.prompt_builder.prompt_builder import PromptBuilder
from src.generators.generator import Generator
from langchain_core.documents import Document
from typing import List
from src.utils.utils import logger
class SimpleRAGPipeLine(PipeLine):
    def __init__(self, retriver: Retriver, prompt_builder: PromptBuilder, generator: Generator):
        super().__init__()
        self.retriver = retriver
        self.prompt_builder = prompt_builder
        self.generator = generator
        
    def ask(self, question: str, k: int=5) -> str:
        if not question.strip():
            logger.error(f"{__class__.__name__}: question is empty")
            raise ValueError("question not be empty")
        result: List[Document] = self.retriver.retrive(question, k=k)
        
        prompt = self.prompt_builder.build_prompt(question=question, docs=result)
        
        answer = self.generator.generate(prompt=prompt)
        return answer
