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
