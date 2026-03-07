"""
FaceMatch class for template-based recognition with cosine similarity, top-k, unknown detection.
"""
import numpy as np
from .similarity import SimilarityMetric

class FaceMatch:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def identify(self, query_embedding, templates):
        scores = {name: SimilarityMetric.cosine_similarity(query_embedding, emb) for name, emb in templates.items()}
        best_name = max(scores, key=scores.get)
        best_score = scores[best_name]
        if best_score >= self.threshold:
            return best_name, best_score
        else:
            return "Unknown", best_score

    def identify_with_scores(self, query_embedding, templates):
        return {name: SimilarityMetric.cosine_similarity(query_embedding, emb) for name, emb in templates.items()}

    def top_k(self, query_embedding, templates, k=5):
        scores = self.identify_with_scores(query_embedding, templates)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:k]
