import json
import random
import uuid
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .config import UPLOAD_DIR
from .db import Base, engine, get_db
from .models import ImageRecord, Book, NgramVocab
from .ocr import run_ocr, detect_page_number_from_regions, extract_page_number
from .schemas import (
    ImageOut,
    BookCreate,
    BookOut,
    ImageAssign,
    TuneRequest,
    TuneResponse,
    TuneCandidate,
)
from .ngram import build_model, serialize_model, deserialize_model, predict

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Page OCR Sorter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=List[ImageOut])
def upload_images(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    records: List[ImageRecord] = []

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        stored_name = f"{uuid.uuid4().hex}{suffix}"
        stored_path = UPLOAD_DIR / stored_name

        with stored_path.open("wb") as f:
            f.write(file.file.read())

        ocr_text = run_ocr(str(stored_path))
        page_number = extract_page_number(ocr_text)
        if page_number is None:
            page_number = detect_page_number_from_regions(str(stored_path))

        record = ImageRecord(
            original_filename=file.filename,
            stored_filename=stored_name,
            page_number=page_number,
            ocr_text=ocr_text,
        )
        db.add(record)
        records.append(record)

    db.commit()
    for record in records:
        db.refresh(record)

    return records


@app.post("/api/upload", response_model=List[ImageOut])
def upload_images_api(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    return upload_images(files=files, db=db)


@app.get("/images", response_model=List[ImageOut])
def list_images(db: Session = Depends(get_db)):
    return db.query(ImageRecord).order_by(ImageRecord.created_at.desc()).all()


@app.get("/api/images", response_model=List[ImageOut])
def list_images_api(db: Session = Depends(get_db)):
    return list_images(db=db)


@app.get("/images/{image_id}/file")
def get_image_file(image_id: int, db: Session = Depends(get_db)):
    record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Image not found")

    path = UPLOAD_DIR / record.stored_filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")

    return FileResponse(path)


@app.post("/books", response_model=BookOut)
def create_book(payload: BookCreate, db: Session = Depends(get_db)):
    book = Book(title=payload.title, author_name=payload.author_name)
    db.add(book)
    db.commit()
    db.refresh(book)
    return book


def get_global_vocab(db: Session) -> set:
    entry = db.query(NgramVocab).filter(NgramVocab.id == 1).first()
    if not entry or not entry.vocab_json:
        return set()
    try:
        return set(json.loads(entry.vocab_json))
    except json.JSONDecodeError:
        return set()


def save_global_vocab(db: Session, vocab: set) -> None:
    payload = json.dumps(sorted(vocab))
    entry = db.query(NgramVocab).filter(NgramVocab.id == 1).first()
    if entry is None:
        entry = NgramVocab(id=1, vocab_json=payload)
        db.add(entry)
    else:
        entry.vocab_json = payload
    db.commit()


def get_ngram_config(db: Session) -> dict:
    entry = db.query(NgramVocab).filter(NgramVocab.id == 1).first()
    if not entry or not entry.config_json:
        return {"n_values": [3, 4, 5], "alpha": 1.0}
    try:
        data = json.loads(entry.config_json)
        return {
            "n_values": data.get("n_values", [3, 4, 5]),
            "alpha": float(data.get("alpha", 1.0)),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"n_values": [3, 4, 5], "alpha": 1.0}


def save_ngram_config(db: Session, config: dict) -> None:
    payload = json.dumps(config)
    entry = db.query(NgramVocab).filter(NgramVocab.id == 1).first()
    if entry is None:
        entry = NgramVocab(id=1, config_json=payload)
        db.add(entry)
    else:
        entry.config_json = payload
    db.commit()


@app.post("/api/books", response_model=BookOut)
def create_book_api(payload: BookCreate, db: Session = Depends(get_db)):
    return create_book(payload=payload, db=db)


@app.get("/books", response_model=List[BookOut])
def list_books(db: Session = Depends(get_db)):
    return db.query(Book).order_by(Book.created_at.desc()).all()


@app.get("/api/books", response_model=List[BookOut])
def list_books_api(db: Session = Depends(get_db)):
    return list_books(db=db)


@app.patch("/images/{image_id}", response_model=ImageOut)
def assign_image(image_id: int, payload: ImageAssign, db: Session = Depends(get_db)):
    record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Image not found")
    record.book_id = payload.book_id
    db.commit()
    db.refresh(record)
    return record


@app.patch("/api/images/{image_id}", response_model=ImageOut)
def assign_image_api(image_id: int, payload: ImageAssign, db: Session = Depends(get_db)):
    return assign_image(image_id=image_id, payload=payload, db=db)


@app.post("/books/{book_id}/train", response_model=BookOut)
def train_book(book_id: int, db: Session = Depends(get_db)):
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    images = (
        db.query(ImageRecord)
        .filter(ImageRecord.book_id == book_id)
        .filter(ImageRecord.ocr_text.isnot(None))
        .all()
    )
    if not images:
        raise HTTPException(status_code=400, detail="No OCR text to train on")

    texts = [img.ocr_text for img in images if img.ocr_text]
    config = get_ngram_config(db)
    model = build_model(texts, n_values=config["n_values"])
    book.model_json = serialize_model(model)
    global_vocab = get_global_vocab(db)
    global_vocab.update(model["counts"].keys())
    save_global_vocab(db, global_vocab)
    db.commit()
    db.refresh(book)
    return book


@app.post("/api/books/{book_id}/train", response_model=BookOut)
def train_book_api(book_id: int, db: Session = Depends(get_db)):
    return train_book(book_id=book_id, db=db)


@app.post("/images/{image_id}/predict", response_model=ImageOut)
def predict_image(image_id: int, db: Session = Depends(get_db)):
    record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Image not found")
    if not record.ocr_text:
        raise HTTPException(status_code=400, detail="No OCR text for image")

    books = db.query(Book).filter(Book.model_json.isnot(None)).all()
    models = {book.id: deserialize_model(book.model_json) for book in books}
    config = get_ngram_config(db)
    global_vocab = get_global_vocab(db)
    results = predict(
        record.ocr_text,
        models,
        vocab_override=global_vocab or None,
        alpha=config["alpha"],
    )
    if not results:
        raise HTTPException(status_code=400, detail="No trained models available")

    best = results[0]
    best_book = db.query(Book).filter(Book.id == best["book_id"]).first()
    record.predicted_book_id = best_book.id if best_book else None
    record.predicted_author = best_book.author_name if best_book else None
    record.predicted_score = f"ppl={best['perplexity']:.4f}"
    db.commit()
    db.refresh(record)
    return record


def retrain_all_books(db: Session, n_values: List[int]) -> None:
    books = db.query(Book).all()
    global_vocab = set()
    for book in books:
        images = (
            db.query(ImageRecord)
            .filter(ImageRecord.book_id == book.id)
            .filter(ImageRecord.ocr_text.isnot(None))
            .all()
        )
        texts = [img.ocr_text for img in images if img.ocr_text]
        if not texts:
            continue
        model = build_model(texts, n_values=n_values)
        book.model_json = serialize_model(model)
        global_vocab.update(model["counts"].keys())
    save_global_vocab(db, global_vocab)
    db.commit()


@app.post("/api/ngram/tune", response_model=TuneResponse)
def tune_ngram(payload: TuneRequest, db: Session = Depends(get_db)):
    n_values_options = payload.n_values_options or [[3, 4, 5], [2, 3, 4, 5], [3, 4]]
    alphas = payload.alphas or [0.5, 1.0, 1.5]
    train_ratio = payload.train_ratio or 0.8
    rng = random.Random(payload.random_seed or 42)

    labeled = {}
    books = db.query(Book).all()
    for book in books:
        images = (
            db.query(ImageRecord)
            .filter(ImageRecord.book_id == book.id)
            .filter(ImageRecord.ocr_text.isnot(None))
            .all()
        )
        texts = [img.ocr_text for img in images if img.ocr_text]
        if len(texts) >= 2:
            labeled[book.id] = texts

    if len(labeled) < 2:
        raise HTTPException(status_code=400, detail="Not enough labeled books to tune")

    candidates: List[TuneCandidate] = []
    best_candidate = None

    for n_values in n_values_options:
        for alpha in alphas:
            models = {}
            test_set = []
            for book_id, texts in labeled.items():
                items = texts[:]
                rng.shuffle(items)
                split = max(1, int(len(items) * train_ratio))
                train_texts = items[:split]
                test_texts = items[split:] or items[-1:]
                models[book_id] = build_model(train_texts, n_values=n_values)
                for text in test_texts:
                    test_set.append((book_id, text))

            vocab = set()
            for model in models.values():
                vocab.update(model["counts"].keys())
            vocab_size = max(len(vocab), 1)

            correct = 0
            total = 0
            ppl_sum = 0.0
            for book_id, text in test_set:
                results = predict(text, models, vocab_override=vocab, alpha=alpha)
                if not results:
                    continue
                total += 1
                if results[0]["book_id"] == book_id:
                    correct += 1
                # perplexity for true book
                true = next((r for r in results if r["book_id"] == book_id), None)
                if true:
                    ppl_sum += true["perplexity"]

            accuracy = correct / total if total else 0.0
            avg_ppl = ppl_sum / total if total else float("inf")
            candidate = TuneCandidate(
                n_values=n_values,
                alpha=alpha,
                accuracy=accuracy,
                avg_perplexity=avg_ppl,
                samples=total,
            )
            candidates.append(candidate)
            if best_candidate is None:
                best_candidate = candidate
            else:
                if candidate.accuracy > best_candidate.accuracy:
                    best_candidate = candidate
                elif candidate.accuracy == best_candidate.accuracy and candidate.avg_perplexity < best_candidate.avg_perplexity:
                    best_candidate = candidate

    if best_candidate is None:
        raise HTTPException(status_code=400, detail="Tuning failed")

    save_ngram_config(db, {"n_values": best_candidate.n_values, "alpha": best_candidate.alpha})
    retrain_all_books(db, n_values=best_candidate.n_values)

    return TuneResponse(best=best_candidate, candidates=candidates)


@app.post("/ngram/tune", response_model=TuneResponse)
def tune_ngram_root(payload: TuneRequest, db: Session = Depends(get_db)):
    return tune_ngram(payload=payload, db=db)


@app.post("/api/images/{image_id}/predict", response_model=ImageOut)
def predict_image_api(image_id: int, db: Session = Depends(get_db)):
    return predict_image(image_id=image_id, db=db)


@app.get("/api/images/{image_id}/file")
def get_image_file_api(image_id: int, db: Session = Depends(get_db)):
    return get_image_file(image_id=image_id, db=db)
