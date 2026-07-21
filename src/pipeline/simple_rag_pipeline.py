from src.pipeline.pipeline import PipeLine
from src.retriver.retriver_base import Retriver 
from src.prompt_builder.prompt_builder import PromptBuilder
from src.generators.generator import Generator
from src.fusion.base_fusion import BaseFusion
from src.fusion.rrffusion import RRFFusion
from src.schemas.retrived_document import RetrivedDocument
from langchain_core.documents import Document
from typing import List
from src.utils.utils import logger
class SimpleRAGPipeLine(PipeLine):
    def __init__(self, retriver_dense: Retriver, retriver_spars: Retriver, fusion: BaseFusion, prompt_builder: PromptBuilder, generator: Generator):
        super().__init__()
        self.retriver_dense = retriver_dense
        self.retriver_sparse = retriver_spars
        self.fusion = fusion
        self.prompt_builder = prompt_builder
        self.generator = generator
        
    def ask(self, question: str, k: int=5) -> str:
        if not question.strip():
            logger.error(f"{__class__.__name__}: question is empty")
            raise ValueError("question not be empty")
        
        # result: List[Document] = self.retriver.retrive(question, k=k)
        result_dense: List[RetrivedDocument] = self.retriver_dense.retrive(question, k=k)
        result_spars: List[RetrivedDocument] = self.retriver_sparse.retrive(question, k=k)
        
        result: List[RetrivedDocument] = self.fusion.merge(dense_results=result_dense, sparse_results=result_spars, k=k)
        
        prompt = self.prompt_builder.build_prompt(question=question, docs=result)
        
        answer = self.generator.generate(prompt=prompt)
        return answer
