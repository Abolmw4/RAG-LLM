from langchain_core.documents import Document
from src.data_ingestion import DataIngestion
from src.chunker import Chunker
from src.embeddings import Embedding
from src.vector_store.vector_store_faiss import FaissVectorStore
from src.prompt_builder.simple_prompt import SimplePrompt
from src.retriver.retriver import SimpleRetriver
from src.generators.ollama_generator import OllamaGenerator
from src.pipeline.simple_rag_pipeline import SimpleRAGPipeLine
from typing import List


def main():
    data_ingst = DataIngestion(path="/home/abolfazl/Documents/rag-llm-project/docs")
    chunker = Chunker(chunk_size=500, chunk_overlap=50)
    embeding = Embedding(model_name="BAAI/bge-m3", device="cuda")
    vectore_store = FaissVectorStore(dimention=1024)    
    prompt_builder = SimplePrompt()
    retriver = SimpleRetriver(embedding=embeding, vector_store=vectore_store)
    generator = OllamaGenerator()
    pipeline = SimpleRAGPipeLine()
    docs: List[Document] = data_ingst.load_documents()
    
    
    
if __name__ == "__main__":
    main()