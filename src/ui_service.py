from __future__ import annotations

import inspect
import time
from dataclasses import asdict, dataclass, field
import os
from src.utils.utils import load_yaml_config_file
from typing import Any, List, Dict


@dataclass
class RAGResponse:
    query: str
    answer: str
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    reranked_documents: List[Dict[str, Any]] = field(default_factory=list)
    prompt: str | None = None
    latency: Dict[str, float] = field(default_factory=dict)
    raw_result: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UIService:
    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.cfg = load_yaml_config_file(config_path)
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Any:
        # اگر اسم کلاس/فایل pipeline تو فرق دارد، فقط این import را اصلاح کن.
        from src.pipeline.simple_rag_pipeline import SimpleRAGPipeLine
        from src.retriver.dense_retriver import DenseRetriver
        from src.retriver.sparse_retriver import SparseRetriver
        from src.fusion.rrffusion import RRFFusion
        from src.embeddings import Embedding
        from src.data_ingestion import DataIngestion
        from src.chunker import Chunker
        from src.vector_store.vector_store_faiss import FaissVectorStore
        from langchain_core.documents import Document
        import numpy as np
        from src.retriver.retriver_base import Retriver
        from src.fusion.base_fusion import BaseFusion
        from src.generators.ollama_generator import OllamaGenerator
        from src.prompt_builder.simple_prompt import SimplePrompt
        
        try:
            embeding = Embedding(model_name=self.cfg.get("Embedding").get("model_name", "BAAI/bge-m3"), device=self.cfg.get("Embedding").get("device", "cuda"))
            if not os.path.exists(self.cfg.get("FIASSVectoreStore").get("index_path")) or not os.path.exists(self.cfg.get("FIASSVectoreStore").get("document_path")):
                data_ingst = DataIngestion(path=self.cfg.get("DataIngestion").get("document_src_dir", "docs"))
                chunker = Chunker(chunk_size=self.cfg.get("Chunker").get("chunk_size"), chunk_overlap=self.cfg.get("Chunker").get("chunk_overlap"))
                vector_store = FaissVectorStore(dimention=self.cfg.get("FIASSVectoreStore").get("dimention"))

                docs: List[Document] = data_ingst.load_documents()
                chunks: List[Document] = chunker.split_documents(documents=docs)
                chunk_embedding: np.ndarray = embeding.embed_documents(chunks=chunks, batch_size=self.cfg.get("Embedding").get("batch_size"),
                                                                       normalize_embeddings=self.cfg.get("Embedding").get("normalize_embeddings"),
                                                                       show_progress_bar=self.cfg.get("Embedding").get("show_progress_bar"), 
                                                                       convert_to_numpy=self.cfg.get("Embedding").get("convert_to_numpy"))
                vector_store.add(embeddings=chunk_embedding, docs=chunks)
                os.makedirs("data", exist_ok=True)
                vector_store.save(index_path=self.cfg.get("FIASSVectoreStore").get("index_path"), document_path=self.cfg.get("FIASSVectoreStore").get("document_path"))

            vector_store: FaissVectorStore = FaissVectorStore.load(index_path=self.cfg.get("FIASSVectoreStore").get("index_path"), document_path=self.cfg.get("FIASSVectoreStore").get("document_path"))
            docs = vector_store.documents

            retriver_dense: Retriver = DenseRetriver(embedding=embeding, vector_store=vector_store)
            retriver_sparse: Retriver = SparseRetriver(documents=docs)
            fusion: BaseFusion = RRFFusion()

            generator = OllamaGenerator()
            prompt_builder = SimplePrompt()
            return SimpleRAGPipeLine(retriver_dense=retriver_dense, retriver_spars=retriver_sparse, fusion=fusion, prompt_builder=prompt_builder, generator=generator)
        except RuntimeError as error:
            print(f"When creating pipeline error occured -> {error}")
            return SimpleRAGPipeLine()

    def ask(self, query: str, chat_history: List[dict[str, str]] | None = None, options: Dict[str, Any] | None = None,) -> Dict[str, Any]:
        chat_history = chat_history or []
        options = options or {}

        start_time = time.perf_counter()
        raw_result = self._call_pipeline(query=query, chat_history=chat_history, options=options,)
        total_time = time.perf_counter() - start_time

        response = self._normalize_result(query=query, raw_result=raw_result, total_time=total_time,)
        return response.to_dict()

    def _call_pipeline(self, query: str, chat_history: List[Dict[str, str]], options: Dict[str, Any],) -> Any:
        # اینجا تلاش می‌کنیم بدون وابستگی شدید به اسم متد pipeline، آن را صدا بزنیم.
        candidate_methods = ["run", "ask", "invoke", "chat", "generate"]

        for method_name in candidate_methods:
            method = getattr(self.pipeline, method_name, None)
            if callable(method):
                return self._invoke_with_supported_args(method=method, query=query, chat_history=chat_history, options=options)

        raise AttributeError(
            "No supported pipeline method found. Expected one of: "
            "run, ask, invoke, chat, generate"
        )

    def _invoke_with_supported_args(self, method: Any, query: str, chat_history: List[Dict[str, str]], options: Dict[str, Any],) -> Any:
        signature = inspect.signature(method)
        kwargs: Dict[str, Any] = {}

        param_names = set(signature.parameters.keys())

        if "query" in param_names:
            kwargs["query"] = query
        elif "question" in param_names:
            kwargs["question"] = query
        elif "user_query" in param_names:
            kwargs["user_query"] = query

        if "chat_history" in param_names:
            kwargs["chat_history"] = chat_history
        elif "history" in param_names:
            kwargs["history"] = chat_history

        if "options" in param_names:
            kwargs["options"] = options
        else:
            for key, value in options.items():
                if key in param_names:
                    kwargs[key] = value

        if not kwargs:
            return method(query)

        return method(**kwargs)

    def _normalize_result(
        self,
        query: str,
        raw_result: Any,
        total_time: float,
    ) -> RAGResponse:
        if isinstance(raw_result, str):
            return RAGResponse(
                query=query,
                answer=raw_result,
                latency={"total": round(total_time, 4)},
                raw_result=raw_result,
            )

        if isinstance(raw_result, dict):
            answer = (
                raw_result.get("answer")
                or raw_result.get("response")
                or raw_result.get("result")
                or raw_result.get("output")
                or ""
            )

            retrieved_documents = (
                raw_result.get("retrieved_documents")
                or raw_result.get("documents")
                or raw_result.get("sources")
                or []
            )

            reranked_documents = raw_result.get("reranked_documents") or []
            prompt = raw_result.get("prompt")

            latency = raw_result.get("latency") or {}
            if "total" not in latency:
                latency["total"] = round(total_time, 4)

            return RAGResponse(
                query=query,
                answer=answer,
                retrieved_documents=self._serialize_docs(retrieved_documents),
                reranked_documents=self._serialize_docs(reranked_documents),
                prompt=prompt,
                latency=latency,
                raw_result=raw_result,
            )

        return RAGResponse(
            query=query,
            answer=str(raw_result),
            latency={"total": round(total_time, 4)},
            raw_result=raw_result,
        )

    def _serialize_docs(self, docs: list[Any]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []

        for index, doc in enumerate(docs, start=1):
            if isinstance(doc, dict):
                item = dict(doc)
                item.setdefault("rank", index)
                serialized.append(item)
                continue

            item = {"rank": index}

            for attr in ["content", "text", "page_content", "document", "score", "source"]:
                if hasattr(doc, attr):
                    item[attr] = getattr(doc, attr)

            if len(item) == 1:
                item["text"] = str(doc)

            serialized.append(item)

        return serialized
