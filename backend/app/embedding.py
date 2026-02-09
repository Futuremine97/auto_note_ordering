import hashlib
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from .config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_LEARNING_RATE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_TRAIN_EPOCHS,
)
from .ngram import extract_ngrams

_sentence_model = None


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


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer

        _sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _sentence_model


def _texts_fingerprint(texts: List[str]) -> str:
    hasher = hashlib.md5()
    for text in texts:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _scale_coords(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return coords.astype(np.float32)
    max_abs = np.max(np.abs(coords), axis=0)
    max_abs[max_abs == 0] = 1.0
    return (coords / max_abs).astype(np.float32)


def _build_cache_path(key: str) -> Path:
    cache_dir = Path(EMBEDDING_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"embedding_{key}.npz"


def _cache_key(
    texts: List[str],
    loss_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> str:
    fingerprint = _texts_fingerprint(texts)
    raw = f"{EMBEDDING_MODEL_NAME}|{loss_type}|{epochs}|{batch_size}|{learning_rate}|{fingerprint}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _encode_texts(texts: List[str]) -> np.ndarray:
    model = _get_sentence_model()
    safe_texts = [text if text else " " for text in texts]
    embeddings = model.encode(
        safe_texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _train_autoencoder(
    embeddings: np.ndarray,
    latent_dim: int,
    loss_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> np.ndarray:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if embeddings.size == 0:
        return np.zeros((0, latent_dim), dtype=np.float32)

    input_dim = embeddings.shape[1]
    hidden_1 = max(64, min(256, input_dim * 2))
    hidden_2 = max(32, min(128, input_dim))

    class AutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_1),
                nn.ReLU(),
                nn.Linear(hidden_1, hidden_2),
                nn.ReLU(),
                nn.Linear(hidden_2, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_2),
                nn.ReLU(),
                nn.Linear(hidden_2, hidden_1),
                nn.ReLU(),
                nn.Linear(hidden_1, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            return recon, z

    device = torch.device("cpu")
    tensor = torch.from_numpy(embeddings).float().to(device)
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True)
    std[std == 0] = 1.0
    tensor = (tensor - mean) / std

    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = AutoEncoder().to(device)
    criterion = nn.L1Loss() if loss_type == "l1" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(max(1, epochs)):
        for (batch,) in loader:
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        _, latents = model(tensor)
    return latents.cpu().numpy().astype(np.float32)


def build_embedding_variant(
    texts: List[str],
    loss_type: str = "l2",
    epochs: int = EMBEDDING_TRAIN_EPOCHS,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    learning_rate: float = EMBEDDING_LEARNING_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    loss_type = (loss_type or "l2").lower()
    key = _cache_key(texts, loss_type, epochs, batch_size, learning_rate)
    cache_path = _build_cache_path(key)

    if cache_path.exists():
        data = np.load(cache_path)
        coords3d = data["coords3d"]
        coords2d = data["coords2d"]
        return coords3d.astype(np.float32), coords2d.astype(np.float32)

    embeddings = _encode_texts(texts)
    latents = _train_autoencoder(
        embeddings,
        latent_dim=3,
        loss_type=loss_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    coords3d = _scale_coords(latents)
    coords2d = _scale_coords(coords3d[:, :2])

    np.savez(cache_path, coords3d=coords3d, coords2d=coords2d)
    return coords3d, coords2d
