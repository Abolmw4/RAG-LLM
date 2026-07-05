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
import numpy as np
import os

def main():
    
    embeding = Embedding(model_name="BAAI/bge-m3", device="cuda")
    
    if not os.path.exists('data/doc.index') or not os.path.exists("data/doc.pkl"):
        print("not found any data")
        data_ingst = DataIngestion(path="/home/abolfazl/Documents/rag-llm-project/docs")
        chunker = Chunker(chunk_size=500, chunk_overlap=50)
        vector_store = FaissVectorStore(dimention=1024)

        docs: List[Document] = data_ingst.load_documents()
        chunks: List[Document] = chunker.split_documents(documents=docs)
        chunk_embedding: np.ndarray = embeding.embed_documents(chunks=chunks)
        vector_store.add(embeddings=chunk_embedding, docs=chunks)
        os.makedirs("data", exist_ok=True)
        vector_store.save(index_path="data/doc.index", document_path="data/doc.pkl")
    
    vector_store: FaissVectorStore = FaissVectorStore.load(index_path='data/doc.index', document_path='data/doc.pkl')

    retriver = SimpleRetriver(embedding=embeding, vector_store=vector_store)
    generator = OllamaGenerator()
    prompt_builder = SimplePrompt()
    pipeline = SimpleRAGPipeLine(retriver=retriver, prompt_builder=prompt_builder, generator=generator)
    
    while True:
        question = input(">> ")

        if question == "خارج شو از چت":
            break

        answer = pipeline.ask(question=question, k=10)
        print(answer)
    
if __name__ == "__main__":
    main()
