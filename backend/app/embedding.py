import hashlib
from typing import Iterable, List

import numpy as np

from .ngram import extract_ngrams


def _stable_hash(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16)


def build_feature_matrix(
    texts: Iterable[str], n_values: List[int], dim: int = 128
) -> np.ndarray:
    if not isinstance(texts, list):
        texts = list(texts)
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        if not text:
            continue
        grams = extract_ngrams(text, n_values)
        for gram in grams:
            idx = _stable_hash(gram) % dim
            matrix[row, idx] += 1.0
    matrix = np.log1p(matrix)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def reduce_to_3d(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return np.zeros((centered.shape[0], 3), dtype=np.float32)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:3].T
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), constant_values=0)
    max_abs = np.max(np.abs(coords), axis=0)
    max_abs[max_abs == 0] = 1.0
    coords = coords / max_abs
    return coords.astype(np.float32)


def build_embeddings_3d(
    texts: List[str], n_values: List[int], dim: int = 128
) -> List[List[float]]:
    matrix = build_feature_matrix(texts, n_values=n_values, dim=dim)
    coords = reduce_to_3d(matrix)
    return coords.tolist()
