from collections import defaultdict
from typing import List

from src.fusion.base_fusion import BaseFusion
from src.schemas.retrived_document import RetrivedDocument


class RRFFusion(BaseFusion):
    def __init__(self, constant: int = 60):
        self.constant = constant

    def merge(self, dense_results: List[RetrivedDocument], sparse_results: List[RetrivedDocument], k: int = 5) -> List[RetrivedDocument]:

        fused_scores = defaultdict(float)
        documents = {}

        # Dense ranking
        for rank, item in enumerate(dense_results, start=1):
            fused_scores[item.doc_id] += 1.0 / (self.constant + rank)
            documents[item.doc_id] = item

        # Sparse ranking
        for rank, item in enumerate(sparse_results, start=1):
            fused_scores[item.doc_id] += 1.0 / (self.constant + rank)
            if item.doc_id not in documents:
                documents[item.doc_id] = item

        ranked_doc_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)

        results = [documents[doc_id] for doc_id in ranked_doc_ids[:k]]

        return results
