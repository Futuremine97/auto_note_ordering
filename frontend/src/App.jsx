import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

function groupByPage(records) {
  const groups = new Map();
  for (const record of records) {
    const key = record.page_number ?? "미인식";
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(record);
  }
  return [...groups.entries()].sort((a, b) => {
    if (a[0] === "미인식") return 1;
    if (b[0] === "미인식") return -1;
    return Number(a[0]) - Number(b[0]);
  });
}

export default function App() {
  const [records, setRecords] = useState([]);
  const [books, setBooks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [bookForm, setBookForm] = useState({ title: "", author_name: "" });
  const [actionMessage, setActionMessage] = useState("");

  const grouped = useMemo(() => groupByPage(records), [records]);

  async function fetchRecords() {
    const res = await fetch(`${API_BASE}/images`);
    if (!res.ok) {
      throw new Error("목록을 불러오지 못했습니다.");
    }
    const data = await res.json();
    setRecords(data);
  }

  async function fetchBooks() {
    const res = await fetch(`${API_BASE}/books`);
    if (!res.ok) {
      throw new Error("책 목록을 불러오지 못했습니다.");
    }
    const data = await res.json();
    setBooks(data);
  }

  useEffect(() => {
    Promise.all([fetchRecords(), fetchBooks()]).catch((err) =>
      setError(err.message)
    );
  }, []);

  function resolveBookName(bookId) {
    const book = books.find((item) => item.id === bookId);
    if (!book) return "미분류";
    return `${book.title} · ${book.author_name}`;
  }

  async function handleCreateBook(event) {
    event.preventDefault();
    if (!bookForm.title.trim() || !bookForm.author_name.trim()) {
      setError("책 제목과 저자명을 입력해주세요.");
      return;
    }
    setError("");
    setActionMessage("");
    const res = await fetch(`${API_BASE}/books`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(bookForm),
    });
    if (!res.ok) {
      setError("책 등록에 실패했습니다.");
      return;
    }
    setBookForm({ title: "", author_name: "" });
    await fetchBooks();
  }

  async function handleAssignBook(imageId, bookId) {
    setActionMessage("");
    const res = await fetch(`${API_BASE}/images/${imageId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ book_id: bookId ? Number(bookId) : null }),
    });
    if (!res.ok) {
      setError("책 라벨 지정에 실패했습니다.");
      return;
    }
    await fetchRecords();
  }

  async function handleTrain(bookId) {
    setActionMessage("");
    const res = await fetch(`${API_BASE}/books/${bookId}/train`, {
      method: "POST",
    });
    if (!res.ok) {
      const message = await res.text();
      setError(message || "모델 학습에 실패했습니다.");
      return;
    }
    setActionMessage("모델 학습이 완료되었습니다.");
  }

  async function handlePredict(imageId) {
    setActionMessage("");
    const res = await fetch(`${API_BASE}/images/${imageId}/predict`, {
      method: "POST",
    });
    if (!res.ok) {
      const message = await res.text();
      setError(message || "예측에 실패했습니다.");
      return;
    }
    await fetchRecords();
    setActionMessage("예측이 완료되었습니다.");
  }

  async function handleUpload(event) {
    const files = event.target.files;
    if (!files.length) return;

    setLoading(true);
    setError("");

    const formData = new FormData();
    for (const file of files) {
      formData.append("files", file);
    }

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const message = await res.text();
        throw new Error(message || "업로드 실패");
      }
      await fetchRecords();
      event.target.value = "";
    } catch (err) {
      setError(err.message || "업로드 실패");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header>
        <div>
          <p className="eyebrow">Page OCR Sorter</p>
          <h1>책 스캔 사진을 페이지 번호로 자동 분류</h1>
          <p className="subtitle">
            이미지를 업로드하면 OCR로 페이지 번호를 인식하고 자동으로 그룹화합니다.
          </p>
        </div>
        <label className={`upload ${loading ? "disabled" : ""}`}>
          <input type="file" multiple accept="image/*" onChange={handleUpload} disabled={loading} />
          {loading ? "처리 중..." : "이미지 업로드"}
        </label>
      </header>

      {error && <div className="error">{error}</div>}
      {actionMessage && <div className="success">{actionMessage}</div>}

      <section className="panel">
        <div>
          <h2>책/저자 등록</h2>
          <p className="muted">OCR 텍스트를 책별로 분류하기 위해 먼저 책을 등록하세요.</p>
        </div>
        <form className="book-form" onSubmit={handleCreateBook}>
          <input
            type="text"
            placeholder="책 제목"
            value={bookForm.title}
            onChange={(event) =>
              setBookForm((prev) => ({ ...prev, title: event.target.value }))
            }
          />
          <input
            type="text"
            placeholder="저자명"
            value={bookForm.author_name}
            onChange={(event) =>
              setBookForm((prev) => ({ ...prev, author_name: event.target.value }))
            }
          />
          <button type="submit">등록</button>
        </form>
        <div className="book-list">
          {books.length === 0 && <span className="muted">등록된 책이 없습니다.</span>}
          {books.map((book) => (
            <div key={book.id} className="book-item">
              <div>
                <strong>{book.title}</strong>
                <span>{book.author_name}</span>
              </div>
              <button type="button" onClick={() => handleTrain(book.id)}>
                모델 학습
              </button>
            </div>
          ))}
        </div>
      </section>

      <section className="grid">
        {grouped.length === 0 && (
          <div className="empty">아직 업로드된 이미지가 없습니다.</div>
        )}

        {grouped.map(([page, items]) => (
          <div key={page} className="card">
            <div className="card-header">
              <span className="tag">페이지 {page}</span>
              <span className="count">{items.length}장</span>
            </div>
            <div className="thumbs">
              {items.map((item) => (
                <figure key={item.id}>
                  <img
                    src={`${API_BASE}/images/${item.id}/file`}
                    alt={item.original_filename}
                    loading="lazy"
                  />
                  <figcaption>{item.original_filename}</figcaption>
                  <div className="meta">
                    <span>라벨: {resolveBookName(item.book_id)}</span>
                    <span>
                      예측:{" "}
                      {item.predicted_book_id
                        ? resolveBookName(item.predicted_book_id)
                        : "미예측"}
                    </span>
                  </div>
                  <div className="actions">
                    <select
                      value={item.book_id || ""}
                      onChange={(event) => handleAssignBook(item.id, event.target.value)}
                    >
                      <option value="">책 선택</option>
                      {books.map((book) => (
                        <option key={book.id} value={book.id}>
                          {book.title} · {book.author_name}
                        </option>
                      ))}
                    </select>
                    <button type="button" onClick={() => handlePredict(item.id)}>
                      예측
                    </button>
                  </div>
                </figure>
              ))}
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}
