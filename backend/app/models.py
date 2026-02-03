from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .db import Base


class ImageRecord(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False, unique=True)
    page_number = Column(Integer, nullable=True, index=True)
    ocr_text = Column(Text, nullable=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=True, index=True)
    predicted_book_id = Column(Integer, ForeignKey("books.id"), nullable=True, index=True)
    predicted_author = Column(String(255), nullable=True)
    predicted_score = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    book = relationship("Book", foreign_keys=[book_id])
    predicted_book = relationship("Book", foreign_keys=[predicted_book_id])


class Book(Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    author_name = Column(String(255), nullable=False)
    model_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class NgramVocab(Base):
    __tablename__ = "ngram_vocab"

    id = Column(Integer, primary_key=True, index=True)
    vocab_json = Column(Text, nullable=True)
    config_json = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
