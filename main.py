from langchain_core.documents import Document
from src.data_ingestion import DataIngestion
from src.chunker import Chunker
from src.embeddings import Embedding
from src.vector_store.vector_store_faiss import FaissVectorStore
from src.prompt_builder.simple_prompt import SimplePrompt
from src.retriver.retriver_base import Retriver
from src.retriver.dense_retriver import DenseRetriver
from src.retriver.sparse_retriver import SparseRetriver
from src.generators.ollama_generator import OllamaGenerator
from src.pipeline.simple_rag_pipeline import SimpleRAGPipeLine
from src.fusion.rrffusion import RRFFusion
from src.fusion.base_fusion import BaseFusion
from typing import List
import numpy as np
import os
from src.utils.utils import load_yaml_config_file, logger
from typing import Dict

def main(cfg: Dict):
    logger.info(f"************ start rag-llm ************")
    embeding = Embedding(model_name=cfg.get("Embedding").get("model_name", "BAAI/bge-m3"), device=cfg.get("Embedding").get("device", "cuda"))
    if not os.path.exists(cfg.get("FIASSVectoreStore").get("index_path")) or not os.path.exists(cfg.get("FIASSVectoreStore").get("document_path")):
        logger.warning("not found any data")
        data_ingst = DataIngestion(path=cfg.get("DataIngestion").get("document_src_dir", "docs"))
        chunker = Chunker(chunk_size=cfg.get("Chunker").get("chunk_size"), chunk_overlap=cfg.get("Chunker").get("chunk_overlap"))
        vector_store = FaissVectorStore(dimention=cfg.get("FIASSVectoreStore").get("dimention"))

        docs: List[Document] = data_ingst.load_documents()
        chunks: List[Document] = chunker.split_documents(documents=docs)
        chunk_embedding: np.ndarray = embeding.embed_documents(chunks=chunks, batch_size=cfg.get("Embedding").get("batch_size"), 
                                                               normalize_embeddings=cfg.get("Embedding").get("normalize_embeddings"), 
                                                               show_progress_bar=cfg.get("Embedding").get("show_progress_bar"), 
                                                               convert_to_numpy=cfg.get("Embedding").get("convert_to_numpy"))
        vector_store.add(embeddings=chunk_embedding, docs=chunks)
        os.makedirs("data", exist_ok=True)
        vector_store.save(index_path=cfg.get("FIASSVectoreStore").get("index_path"), document_path=cfg.get("FIASSVectoreStore").get("document_path"))
    vector_store: FaissVectorStore = FaissVectorStore.load(index_path=cfg.get("FIASSVectoreStore").get("index_path"), document_path=cfg.get("FIASSVectoreStore").get("document_path"))
    docs = vector_store.documents
    # retriver: Retriver = SimpleRetriver(embedding=embeding, vector_store=vector_store)
    retriver_dense: Retriver = DenseRetriver(embedding=embeding, vector_store=vector_store)
    retriver_sparse: Retriver = SparseRetriver(documents=docs)
    fusion: BaseFusion = RRFFusion()
    
    generator = OllamaGenerator()
    prompt_builder = SimplePrompt()
    # pipeline = SimpleRAGPipeLine(retriver=retriver, prompt_builder=prompt_builder, generator=generator)
    pipeline = SimpleRAGPipeLine(retriver_dense=retriver_dense, retriver_spars=retriver_sparse, fusion=fusion, prompt_builder=prompt_builder, generator=generator)
    
    while True:
        question = input(">> ")

        if question == "خارج شو از چت":
            break

        answer = pipeline.ask(question=question, k=10)
        print(answer)
    
if __name__ == "__main__":
    configs = load_yaml_config_file(config_src="configs/config.yaml")
    main(configs)
