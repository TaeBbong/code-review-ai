"""
Semantic matching for evaluation using sentence embeddings.

Uses lazy loading to avoid startup overhead - the embedding model
is only loaded when first needed.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class SemanticMatcher:
    """
    Semantic similarity matcher using sentence-transformers.

    Uses lazy loading - model is only loaded on first use.
    Singleton pattern ensures model is loaded only once.
    """

    _instance: Optional["SemanticMatcher"] = None
    _model = None

    # Default model - small and fast, good for semantic similarity
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __new__(cls) -> "SemanticMatcher":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_model_loaded(self) -> None:
        """Load the model if not already loaded."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.DEFAULT_MODEL)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of texts to encode

        Returns:
            Numpy array of embeddings, shape (len(texts), embedding_dim)
        """
        self._ensure_model_loaded()
        return self._model.encode(texts, convert_to_numpy=True)

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between 0 and 1
        """
        embeddings = self.encode([text1, text2])
        # Cosine similarity
        sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(sim)

    def similarity_to_any(
        self,
        text: str,
        candidates: list[str],
        threshold: float = 0.5,
    ) -> tuple[bool, float, Optional[str]]:
        """
        Check if text is semantically similar to any of the candidates.

        Args:
            text: Text to check
            candidates: List of candidate texts to compare against
            threshold: Minimum similarity score to consider a match

        Returns:
            Tuple of (matched, best_score, best_candidate)
        """
        if not candidates:
            return True, 1.0, None  # No candidates = no constraint

        if not text.strip():
            return False, 0.0, None

        # Encode all at once for efficiency
        all_texts = [text] + candidates
        embeddings = self.encode(all_texts)

        text_emb = embeddings[0]
        candidate_embs = embeddings[1:]

        # Calculate similarities
        best_score = 0.0
        best_candidate = None

        for i, cand_emb in enumerate(candidate_embs):
            sim = np.dot(text_emb, cand_emb) / (
                np.linalg.norm(text_emb) * np.linalg.norm(cand_emb)
            )
            if sim > best_score:
                best_score = float(sim)
                best_candidate = candidates[i]

        return best_score >= threshold, best_score, best_candidate


# Global instance for convenience
_matcher: Optional[SemanticMatcher] = None


def get_semantic_matcher() -> SemanticMatcher:
    """Get the global SemanticMatcher instance."""
    global _matcher
    if _matcher is None:
        _matcher = SemanticMatcher()
    return _matcher


def semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts.

    Convenience function that uses the global matcher.
    """
    return get_semantic_matcher().similarity(text1, text2)


def matches_any_semantically(
    text: str,
    candidates: list[str],
    threshold: float = 0.5,
) -> tuple[bool, float]:
    """
    Check if text semantically matches any candidate.

    Args:
        text: Text to check
        candidates: Candidate texts to compare against
        threshold: Minimum similarity (0-1)

    Returns:
        Tuple of (matched, best_score)
    """
    matched, score, _ = get_semantic_matcher().similarity_to_any(
        text, candidates, threshold
    )
    return matched, score
