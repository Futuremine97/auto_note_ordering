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
    cluster_id: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


class ImageSummary(BaseModel):
    id: int
    original_filename: str
    stored_filename: str
    page_number: Optional[int]
    book_id: Optional[int]
    predicted_book_id: Optional[int]
    predicted_author: Optional[str]
    predicted_score: Optional[str]
    cluster_id: Optional[int]
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


class ClusterRequest(BaseModel):
    n_values: Optional[List[int]] = None
    threshold: Optional[float] = 0.25
    limit: Optional[int] = None
    apply_labels: Optional[bool] = False
    min_ratio: Optional[float] = 0.6
    min_votes: Optional[int] = 2
    overwrite: Optional[bool] = False


class ClusterResponse(BaseModel):
    total: int
    clustered: int
    clusters: int
    applied_labels: int


class EmbeddingRequest(BaseModel):
    n_values: Optional[List[int]] = None
    dim: Optional[int] = 128
    limit: Optional[int] = None


class EmbeddingPoint(BaseModel):
    id: int
    x: float
    y: float
    z: float
    cluster_id: Optional[int]
    book_id: Optional[int]
    predicted_book_id: Optional[int]
    page_number: Optional[int]


class EmbeddingResponse(BaseModel):
    total: int
    points: List[EmbeddingPoint]


class LlmMessage(BaseModel):
    role: str
    content: str


class LlmDiscussRequest(BaseModel):
    prompt: str
    include_all_images: Optional[bool] = True
    include_images: Optional[bool] = False
    max_images: Optional[int] = None
    image_ids: Optional[List[int]] = None
    messages: Optional[List[LlmMessage]] = None


class LlmDiscussResponse(BaseModel):
    answer: str
    used_images: int
    ocr_chars: int
    included_images: int


class AuthLogin(BaseModel):
    password: str


class AuthStatus(BaseModel):
    authenticated: bool


class SpeechRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    response_format: Optional[str] = None
    speed: Optional[float] = None
