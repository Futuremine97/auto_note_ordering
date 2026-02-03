import json
import math
from collections import Counter
from typing import Dict, Iterable, List


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


def predict(text: str, models_by_id: Dict[int, Dict]) -> List[Dict]:
    if not models_by_id:
        return []

    # Build shared vocabulary for comparable smoothing.
    vocab = set()
    for model in models_by_id.values():
        vocab.update(model["counts"].keys())
    vocab_size = max(len(vocab), 1)

    results = []
    for book_id, model in models_by_id.items():
        n_values = model["n_values"]
        grams = extract_ngrams(text, n_values)
        if not grams:
            score = float("-inf")
        else:
            total = model["total"]
            counts = model["counts"]
            alpha = 1.0
            denom = total + alpha * vocab_size
            score = 0.0
            for gram in grams:
                score += math.log((counts.get(gram, 0) + alpha) / denom)
        results.append({"book_id": book_id, "score": score})

    results.sort(key=lambda item: item["score"], reverse=True)
    return results
