import json
import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def extract_ngrams(text: str, n_values: Iterable[int]) -> List[str]:
    text = normalize_text(text)
    ngrams: List[str] = []
    for n in n_values:
        if len(text) < n:
            continue
        for i in range(len(text) - n + 1):
            ngrams.append(text[i : i + n])
    return ngrams


def build_model(texts: List[str], n_values: Iterable[int]) -> Dict:
    counts = Counter()
    total = 0
    for text in texts:
        grams = extract_ngrams(text, n_values)
        counts.update(grams)
        total += len(grams)
    return {
        "n_values": list(n_values),
        "total": total,
        "counts": counts,
    }


def serialize_model(model: Dict) -> str:
    payload = {
        "n_values": model["n_values"],
        "total": model["total"],
        "counts": dict(model["counts"]),
    }
    return json.dumps(payload)


def deserialize_model(model_json: str) -> Dict:
    data = json.loads(model_json)
    return {
        "n_values": data["n_values"],
        "total": data["total"],
        "counts": Counter(data["counts"]),
    }


def _perplexity(
    text: str, model: Dict, vocab_size: int, alpha: float
) -> float:
    grams = extract_ngrams(text, model["n_values"])
    if not grams:
        return float("inf")
    total = model["total"]
    counts = model["counts"]
    denom = total + alpha * vocab_size
    log_prob = 0.0
    for gram in grams:
        log_prob += math.log((counts.get(gram, 0) + alpha) / denom)
    avg_log_prob = log_prob / len(grams)
    return math.exp(-avg_log_prob)


def predict(
    text: str,
    models_by_id: Dict[int, Dict],
    vocab_override: Optional[Set[str]] = None,
    alpha: float = 1.0,
) -> List[Dict]:
    if not models_by_id:
        return []

    # Build shared vocabulary for comparable smoothing (or use provided global vocab).
    if vocab_override is None:
        vocab = set()
        for model in models_by_id.values():
            vocab.update(model["counts"].keys())
    else:
        vocab = vocab_override
    vocab_size = max(len(vocab), 1)

    results = []
    for book_id, model in models_by_id.items():
        ppl = _perplexity(text, model, vocab_size, alpha)
        results.append({"book_id": book_id, "perplexity": ppl})

    results.sort(key=lambda item: item["perplexity"])
    return results
