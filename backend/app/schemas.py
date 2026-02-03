from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class ImageOut(BaseModel):
    id: int
    original_filename: str
    stored_filename: str
    page_number: Optional[int]
    ocr_text: Optional[str]
    book_id: Optional[int]
    predicted_book_id: Optional[int]
    predicted_author: Optional[str]
    predicted_score: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class BookCreate(BaseModel):
    title: str
    author_name: str


class BookOut(BaseModel):
    id: int
    title: str
    author_name: str
    model_json: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ImageAssign(BaseModel):
    book_id: Optional[int]


class TuneRequest(BaseModel):
    n_values_options: Optional[List[List[int]]] = None
    alphas: Optional[List[float]] = None
    train_ratio: Optional[float] = 0.8
    random_seed: Optional[int] = 42


class TuneCandidate(BaseModel):
    n_values: List[int]
    alpha: float
    accuracy: float
    avg_perplexity: float
    samples: int


class TuneResponse(BaseModel):
    best: TuneCandidate
    candidates: List[TuneCandidate]


class BulkPredictRequest(BaseModel):
    only_unlabeled: Optional[bool] = True
    apply_labels: Optional[bool] = False
    min_ratio: Optional[float] = 1.15
    limit: Optional[int] = None


class BulkPredictResponse(BaseModel):
    total: int
    predicted: int
    applied_labels: int
