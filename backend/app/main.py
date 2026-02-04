import io
import json
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from PIL import Image

from .config import UPLOAD_DIR, OCR_WORKERS
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
    BulkPredictRequest,
    BulkPredictResponse,
    ClusterRequest,
    ClusterResponse,
    LlmDiscussRequest,
    LlmDiscussResponse,
    AuthLogin,
    AuthStatus,
)
from .ngram import (
    build_model,
    serialize_model,
    deserialize_model,
    predict,
    build_symbol_vocab,
    extract_ngrams,
)
from .llm import build_ocr_context, build_image_payloads, call_llm
from .auth import AUTH_COOKIE_NAME, create_auth_cookie, require_auth, verify_auth_cookie
from .config import AUTH_COOKIE_SECURE, AUTH_COOKIE_TTL_HOURS, PHOTO_PASSWORD

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


@app.get("/api/auth/status", response_model=AuthStatus)
def auth_status(request: Request):
    if not PHOTO_PASSWORD:
        return AuthStatus(authenticated=True)
    token = request.cookies.get(AUTH_COOKIE_NAME)
    return AuthStatus(authenticated=bool(token and verify_auth_cookie(token)))


@app.post("/api/auth/login", response_model=AuthStatus)
def auth_login(payload: AuthLogin, response: Response):
    if not PHOTO_PASSWORD:
        return AuthStatus(authenticated=True)
    if payload.password != PHOTO_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    token = create_auth_cookie()
    response.set_cookie(
        AUTH_COOKIE_NAME,
        token,
        httponly=True,
        secure=AUTH_COOKIE_SECURE,
        samesite="Lax",
        max_age=AUTH_COOKIE_TTL_HOURS * 3600,
    )
    return AuthStatus(authenticated=True)


@app.post("/api/auth/logout", response_model=AuthStatus)
def auth_logout(response: Response):
    response.delete_cookie(AUTH_COOKIE_NAME)
    return AuthStatus(authenticated=False)


@app.post("/upload", response_model=List[ImageOut])
def upload_images(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    records: List[ImageRecord] = []
    pending = []

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix in {".heic", ".heif"}:
            try:
                image = Image.open(io.BytesIO(file.file.read()))
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail="HEIC 파일을 처리할 수 없습니다. JPG/PNG로 변환 후 업로드해주세요.",
                ) from exc
            stored_name = f"{uuid.uuid4().hex}.jpg"
            stored_path = UPLOAD_DIR / stored_name
            image.convert("RGB").save(stored_path, format="JPEG", quality=95)
        else:
            stored_name = f"{uuid.uuid4().hex}{suffix}"
            stored_path = UPLOAD_DIR / stored_name
            with stored_path.open("wb") as f:
                f.write(file.file.read())

        pending.append(
            {
                "original_filename": file.filename,
                "stored_filename": stored_name,
                "stored_path": stored_path,
            }
        )

    def process_one(item):
        ocr_text = run_ocr(str(item["stored_path"]))
        page_number = detect_page_number_from_regions(str(item["stored_path"]))
        if page_number is None:
            page_number = extract_page_number(ocr_text)
        return ocr_text, page_number

    worker_count = max(1, min(OCR_WORKERS, len(pending)))
    results = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(process_one, item): item for item in pending}
        for future in as_completed(future_map):
            item = future_map[future]
            try:
                ocr_text, page_number = future.result()
            except Exception:
                ocr_text, page_number = "", None
            results[item["stored_filename"]] = (ocr_text, page_number)

    for item in pending:
        ocr_text, page_number = results.get(item["stored_filename"], ("", None))
        record = ImageRecord(
            original_filename=item["original_filename"],
            stored_filename=item["stored_filename"],
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
def list_images(request: Request, db: Session = Depends(get_db)):
    require_auth(request)
    return db.query(ImageRecord).order_by(ImageRecord.created_at.desc()).all()


@app.get("/api/images", response_model=List[ImageOut])
def list_images_api(request: Request, db: Session = Depends(get_db)):
    return list_images(request=request, db=db)


@app.get("/images/{image_id}/file")
def get_image_file(request: Request, image_id: int, db: Session = Depends(get_db)):
    require_auth(request)
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


@app.post("/llm/discuss", response_model=LlmDiscussResponse)
def discuss_with_llm(payload: LlmDiscussRequest, db: Session = Depends(get_db)):
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt가 필요합니다.")

    query = db.query(ImageRecord)
    if payload.image_ids:
        query = query.filter(ImageRecord.id.in_(payload.image_ids))
    records = query.all()
    if payload.include_all_images is False and not payload.image_ids:
        records = []

    records.sort(
        key=lambda item: (
            item.page_number is None,
            item.page_number or 0,
            item.id,
        )
    )

    context, used_images = build_ocr_context(records)
    image_payloads = []
    included_images = 0
    if payload.include_images:
        image_payloads, included_images = build_image_payloads(
            records, max_images=payload.max_images
        )
        if included_images == 0:
            raise HTTPException(status_code=400, detail="이미지 전송 대상이 없습니다.")

    if not context and not image_payloads:
        raise HTTPException(status_code=400, detail="OCR 텍스트가 없습니다.")

    messages = payload.messages or []
    system = {
        "role": "system",
        "content": (
            "너는 연구 조교다. 제공된 스캔 OCR을 기반으로 사용자의 연구 질문에 답하고 "
            "필요하면 페이지 번호를 근거로 요약, 비교, 토론 포인트를 제시한다."
        ),
    }
    if image_payloads:
        content = [
            {
                "type": "text",
                "text": f"연구 토픽: {prompt}\n\n[스캔 OCR]\n{context}",
            },
            *image_payloads,
        ]
    else:
        content = f"연구 토픽: {prompt}\n\n[스캔 OCR]\n{context}"
    user = {"role": "user", "content": content}
    assembled = [system, *messages, user]
    try:
        answer = call_llm(assembled)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return LlmDiscussResponse(
        answer=answer,
        used_images=used_images,
        ocr_chars=len(context),
        included_images=included_images,
    )


@app.post("/api/llm/discuss", response_model=LlmDiscussResponse)
def discuss_with_llm_api(payload: LlmDiscussRequest, db: Session = Depends(get_db)):
    return discuss_with_llm(payload=payload, db=db)


def get_global_vocab(db: Session, n_values: Optional[List[int]] = None) -> set:
    entry = db.query(NgramVocab).filter(NgramVocab.id == 1).first()
    if not entry or not entry.vocab_json:
        vocab = set()
    else:
        try:
            vocab = set(json.loads(entry.vocab_json))
        except json.JSONDecodeError:
            vocab = set()
    if n_values:
        vocab.update(build_symbol_vocab(n_values))
    return vocab


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
    global_vocab = get_global_vocab(db, config["n_values"])
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
    global_vocab = get_global_vocab(db, config["n_values"])
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


@app.post("/api/images/predict-all", response_model=BulkPredictResponse)
def predict_all_images(payload: BulkPredictRequest, db: Session = Depends(get_db)):
    books = db.query(Book).filter(Book.model_json.isnot(None)).all()
    if not books:
        raise HTTPException(status_code=400, detail="No trained models available")

    models = {book.id: deserialize_model(book.model_json) for book in books}
    config = get_ngram_config(db)
    global_vocab = get_global_vocab(db, config["n_values"])

    query = db.query(ImageRecord).filter(ImageRecord.ocr_text.isnot(None))
    if payload.only_unlabeled:
        query = query.filter(ImageRecord.book_id.is_(None))
    if payload.limit:
        query = query.limit(payload.limit)
    images = query.all()

    predicted = 0
    applied = 0
    for record in images:
        results = predict(
            record.ocr_text,
            models,
            vocab_override=global_vocab or None,
            alpha=config["alpha"],
        )
        if not results:
            continue
        best = results[0]
        best_book = db.query(Book).filter(Book.id == best["book_id"]).first()
        record.predicted_book_id = best_book.id if best_book else None
        record.predicted_author = best_book.author_name if best_book else None
        record.predicted_score = f"ppl={best['perplexity']:.4f}"
        predicted += 1

        if payload.apply_labels and best_book:
            if len(results) == 1:
                record.book_id = best_book.id
                applied += 1
            else:
                ratio = results[1]["perplexity"] / max(best["perplexity"], 1e-6)
                if ratio >= (payload.min_ratio or 1.15):
                    record.book_id = best_book.id
                    applied += 1

    db.commit()
    return BulkPredictResponse(
        total=len(images),
        predicted=predicted,
        applied_labels=applied,
    )


@app.post("/images/predict-all", response_model=BulkPredictResponse)
def predict_all_images_root(payload: BulkPredictRequest, db: Session = Depends(get_db)):
    return predict_all_images(payload=payload, db=db)


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = a.intersection(b)
    union = a.union(b)
    return len(inter) / max(len(union), 1)


@app.post("/api/images/cluster", response_model=ClusterResponse)
def cluster_images(payload: ClusterRequest, db: Session = Depends(get_db)):
    n_values = payload.n_values or [3, 4, 5]
    threshold = payload.threshold or 0.25
    max_clusters = 10

    query = db.query(ImageRecord).filter(ImageRecord.ocr_text.isnot(None))
    if payload.limit:
        query = query.limit(payload.limit)
    images = query.order_by(ImageRecord.id.asc()).all()

    if not images:
        raise HTTPException(status_code=400, detail="No OCR images to cluster")

    clusters = []
    next_cluster_id = 1
    clustered = 0

    for record in images:
        grams = set(extract_ngrams(record.ocr_text or "", n_values))
        if not grams:
            record.cluster_id = None
            continue

        best_cluster = None
        best_score = 0.0
        for cluster in clusters:
            score = _jaccard(grams, cluster["grams"])
            if score > best_score:
                best_score = score
                best_cluster = cluster

        if best_cluster and best_score >= threshold:
            record.cluster_id = best_cluster["id"]
            best_cluster["grams"].update(grams)
        else:
            if len(clusters) >= max_clusters:
                # Force assignment to closest cluster when max reached.
                if best_cluster:
                    record.cluster_id = best_cluster["id"]
                    best_cluster["grams"].update(grams)
                else:
                    record.cluster_id = clusters[0]["id"]
                    clusters[0]["grams"].update(grams)
                clustered += 1
                continue
            record.cluster_id = next_cluster_id
            clusters.append({"id": next_cluster_id, "grams": set(grams)})
            next_cluster_id += 1

        clustered += 1

    applied = 0
    if payload.apply_labels:
        cluster_map = {}
        for record in images:
            if record.cluster_id is None:
                continue
            cluster_map.setdefault(record.cluster_id, []).append(record)

        for cluster_id, items in cluster_map.items():
            counts = {}
            for item in items:
                if item.predicted_book_id:
                    counts[item.predicted_book_id] = counts.get(item.predicted_book_id, 0) + 1

            if not counts:
                continue

            best_book_id, best_count = max(counts.items(), key=lambda it: it[1])
            ratio = best_count / max(len(items), 1)
            if best_count < (payload.min_votes or 2) or ratio < (payload.min_ratio or 0.6):
                continue

            for item in items:
                if payload.overwrite or item.book_id is None:
                    item.book_id = best_book_id
                    applied += 1

    db.commit()
    return ClusterResponse(
        total=len(images),
        clustered=clustered,
        clusters=len(clusters),
        applied_labels=applied,
    )


@app.post("/images/cluster", response_model=ClusterResponse)
def cluster_images_root(payload: ClusterRequest, db: Session = Depends(get_db)):
    return cluster_images(payload=payload, db=db)


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
    global_vocab.update(build_symbol_vocab(n_values))
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
            vocab.update(build_symbol_vocab(n_values))
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
