import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

function sortByPage(items) {
  return [...items].sort((a, b) => {
    if (a.page_number == null && b.page_number == null) return a.id - b.id;
    if (a.page_number == null) return 1;
    if (b.page_number == null) return -1;
    if (a.page_number === b.page_number) return a.id - b.id;
    return a.page_number - b.page_number;
  });
}

function groupByBook(records, books) {
  const bookLabel = (bookId) => {
    const book = books.find((item) => item.id === bookId);
    if (!book) return "미분류";
    return `${book.title} · ${book.author_name}`;
  };

  const groups = new Map();
  for (const record of records) {
    let key = "미분류";
    if (record.book_id) {
      key = `book_${record.book_id}`;
    } else if (record.predicted_book_id) {
      key = `pred_${record.predicted_book_id}`;
    }
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(record);
  }

  return [...groups.entries()]
    .map(([key, items]) => ({
      key,
      label: key === "미분류"
        ? "미분류"
        : key.startsWith("pred_")
          ? `예측: ${bookLabel(Number(key.replace("pred_", "")))}`
          : bookLabel(Number(key.replace("book_", ""))),
      items: sortByPage(items),
    }))
    .sort((a, b) => {
      if (a.label === "미분류") return 1;
      if (b.label === "미분류") return -1;
      if (a.label.startsWith("예측") && !b.label.startsWith("예측")) return 1;
      if (!a.label.startsWith("예측") && b.label.startsWith("예측")) return -1;
      return a.label.localeCompare(b.label);
    });
}

function groupByCluster(records) {
  const groups = new Map();
  for (const record of records) {
    const key = record.cluster_id ?? "미분류";
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(record);
  }

  return [...groups.entries()]
    .map(([key, items]) => ({
      key,
      label: key === "미분류" ? "미분류" : `클러스터 ${key}`,
      items: sortByPage(items),
    }))
    .sort((a, b) => {
      if (a.label === "미분류") return 1;
      if (b.label === "미분류") return -1;
      const aId = Number(a.key);
      const bId = Number(b.key);
      return aId - bId;
    });
}

export default function App() {
  const [records, setRecords] = useState([]);
  const [books, setBooks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [bookForm, setBookForm] = useState({ title: "", author_name: "" });
  const [actionMessage, setActionMessage] = useState("");
  const [viewMode, setViewMode] = useState("book");
  const [chatMessages, setChatMessages] = useState([]);
  const [chatPrompt, setChatPrompt] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");
  const [includeAllImages, setIncludeAllImages] = useState(true);
  const [includeVision, setIncludeVision] = useState(true);
  const [maxVisionImages, setMaxVisionImages] = useState(12);
  const chatEndRef = useRef(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [isAuthed, setIsAuthed] = useState(false);
  const [passwordInput, setPasswordInput] = useState("");
  const [authError, setAuthError] = useState("");
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authReason, setAuthReason] = useState("");
  const pendingAuthAction = useRef(null);
  const [revealedImages, setRevealedImages] = useState([]);
  const uploadDisabled = loading || (!isAuthed && authChecked);

  const grouped = useMemo(() => {
    return viewMode === "cluster"
      ? groupByCluster(records)
      : groupByBook(records, books);
  }, [records, books, viewMode]);

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
    checkAuth();
  }, []);

  async function checkAuth() {
    try {
      const res = await fetch(`${API_BASE}/auth/status`);
      if (res.ok) {
        const data = await res.json();
        setIsAuthed(Boolean(data.authenticated));
      } else {
        setIsAuthed(false);
      }
    } catch {
      setIsAuthed(false);
    } finally {
      setAuthChecked(true);
    }
  }

  useEffect(() => {
    if (!authChecked) return;
    Promise.all([fetchRecords(), fetchBooks()]).catch((err) =>
      setError(err.message)
    );
  }, [authChecked]);

  useEffect(() => {
    if (!chatEndRef.current) return;
    chatEndRef.current.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, chatLoading]);

  function resolveBookName(bookId) {
    const book = books.find((item) => item.id === bookId);
    if (!book) return "미분류";
    return `${book.title} · ${book.author_name}`;
  }

  function requestAuth(reason, action) {
    if (isAuthed) {
      if (action) action();
      return true;
    }
    pendingAuthAction.current = action || null;
    setAuthReason(reason || "사진을 보려면 비밀번호를 입력해야 합니다.");
    setAuthError("");
    setShowAuthModal(true);
    return false;
  }

  function revealImage(imageId) {
    requestAuth("이미지를 보려면 비밀번호가 필요합니다.", () => {
      setRevealedImages((prev) =>
        prev.includes(imageId) ? prev : [...prev, imageId]
      );
    });
  }

  async function handlePredictAll(applyLabels = false) {
    if (!requestAuth("전체 자동 분류를 진행하려면 비밀번호가 필요합니다.")) {
      return;
    }
    setActionMessage("");
    setError("");
    const res = await fetch(`${API_BASE}/images/predict-all`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        only_unlabeled: true,
        apply_labels: applyLabels,
        min_ratio: 1.2,
      }),
    });
    if (!res.ok) {
      const message = await res.text();
      setError(message || "전체 예측에 실패했습니다.");
      return;
    }
    const data = await res.json();
    if (applyLabels) {
      setActionMessage(
        `전체 예측 완료 (${data.predicted}/${data.total}) · 라벨 적용 ${data.applied_labels}건`
      );
    } else {
      setActionMessage(`전체 예측 완료 (${data.predicted}/${data.total})`);
    }
    await fetchRecords();
  }

  async function handleCluster(applyLabels = false) {
    if (!requestAuth("클러스터링을 진행하려면 비밀번호가 필요합니다.")) {
      return;
    }
    setActionMessage("");
    setError("");
    const res = await fetch(`${API_BASE}/images/cluster`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        threshold: 0.25,
        apply_labels: applyLabels,
        min_ratio: 0.6,
        min_votes: 2,
      }),
    });
    if (!res.ok) {
      const message = await res.text();
      setError(message || "클러스터링에 실패했습니다.");
      return;
    }
    const data = await res.json();
    if (applyLabels) {
      setActionMessage(
        `클러스터링 완료 (${data.clustered}/${data.total}, ${data.clusters}개 그룹) · 라벨 적용 ${data.applied_labels}건`
      );
    } else {
      setActionMessage(
        `클러스터링 완료 (${data.clustered}/${data.total}, ${data.clusters}개 그룹)`
      );
    }
    await fetchRecords();
    setViewMode("cluster");
  }

  async function handleCreateBook(event) {
    event.preventDefault();
    if (!requestAuth("책 등록을 하려면 비밀번호가 필요합니다.")) {
      return;
    }
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
    if (!requestAuth("라벨을 지정하려면 비밀번호가 필요합니다.")) {
      return;
    }
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
    if (!requestAuth("모델 학습을 하려면 비밀번호가 필요합니다.")) {
      return;
    }
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
    if (!requestAuth("예측을 하려면 비밀번호가 필요합니다.")) {
      return;
    }
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
    if (!requestAuth("이미지 업로드를 하려면 비밀번호가 필요합니다.")) {
      event.target.value = "";
      return;
    }
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

  async function handleDiscuss() {
    if (!chatPrompt.trim()) return;
    if (!requestAuth("LLM 대화는 비밀번호 입력 후 이용할 수 있습니다.")) {
      return;
    }
    setChatError("");
    setChatLoading(true);

    const history = [...chatMessages];
    const prompt = chatPrompt.trim();
    setChatPrompt("");
    setChatMessages([...history, { role: "user", content: prompt }]);

    try {
      const res = await fetch(`${API_BASE}/llm/discuss`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          include_all_images: includeAllImages,
          include_images: includeVision,
          max_images: includeVision ? Number(maxVisionImages) || 12 : null,
          messages: history,
        }),
      });
      if (!res.ok) {
        const message = await res.text();
        throw new Error(message || "LLM 요청에 실패했습니다.");
      }
      const data = await res.json();
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer },
      ]);
    } catch (err) {
      setChatError(err.message || "LLM 요청에 실패했습니다.");
    } finally {
      setChatLoading(false);
    }
  }

  async function handleAuthSubmit(event) {
    event.preventDefault();
    setAuthError("");
    if (!passwordInput.trim()) {
      setAuthError("비밀번호를 입력해주세요.");
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password: passwordInput }),
      });
      if (!res.ok) {
        const message = await res.text();
        setAuthError(message || "비밀번호가 올바르지 않습니다.");
        return;
      }
      setPasswordInput("");
      setIsAuthed(true);
      setShowAuthModal(false);
      setAuthReason("");
      await Promise.all([fetchRecords(), fetchBooks()]);
      if (pendingAuthAction.current) {
        const action = pendingAuthAction.current;
        pendingAuthAction.current = null;
        action();
      }
    } catch (err) {
      setAuthError(err.message || "로그인 실패");
    }
  }

  function closeAuthModal() {
    setShowAuthModal(false);
    setAuthError("");
    setPasswordInput("");
    pendingAuthAction.current = null;
  }

  function handleUploadClick(event) {
    if (!isAuthed) {
      event.preventDefault();
      event.stopPropagation();
      requestAuth("이미지 업로드를 하려면 비밀번호가 필요합니다.");
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
        <label
          className={`upload ${uploadDisabled ? "disabled" : ""}`}
          onClick={handleUploadClick}
        >
          <input
            type="file"
            multiple
            accept="image/*"
            onChange={handleUpload}
            disabled={uploadDisabled}
          />
          {loading
            ? "처리 중..."
            : !isAuthed
              ? "로그인 후 업로드"
              : "이미지 업로드"}
        </label>
      </header>

      {error && <div className="error">{error}</div>}
      {actionMessage && <div className="success">{actionMessage}</div>}

      <section className={`panel ${!isAuthed ? "disabled-panel" : ""}`}>
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
        <div className="bulk-actions">
          <button type="button" onClick={() => handlePredictAll(false)}>
            전체 자동 분류(예측)
          </button>
          <button type="button" className="secondary" onClick={() => handlePredictAll(true)}>
            전체 예측 + 라벨 적용
          </button>
        </div>
        <div className="view-actions">
          <button type="button" className="secondary" onClick={() => handleCluster(false)}>
            이미지 클러스터링
          </button>
          <button type="button" onClick={() => handleCluster(true)}>
            클러스터 라벨 적용
          </button>
          <div className="toggle">
            <button
              type="button"
              className={viewMode === "book" ? "active" : ""}
              onClick={() => setViewMode("book")}
            >
              책/예측 기준
            </button>
            <button
              type="button"
              className={viewMode === "cluster" ? "active" : ""}
              onClick={() => setViewMode("cluster")}
            >
              클러스터 기준
            </button>
          </div>
        </div>
      </section>

      <section className={`panel chat-panel ${!isAuthed ? "disabled-panel" : ""}`}>
        <div className="chat-header">
          <div>
            <h2>LLM 연구 대화</h2>
            <p className="muted">
              업로드된 OCR 텍스트를 기반으로 연구 토픽을 논의합니다.
            </p>
          </div>
          <div className="chat-toggles">
            <label className="chat-toggle">
              <input
                type="checkbox"
                checked={includeAllImages}
                onChange={(event) => setIncludeAllImages(event.target.checked)}
              />
              모든 이미지 OCR 포함
            </label>
            <label className="chat-toggle">
              <input
                type="checkbox"
                checked={includeVision}
                onChange={(event) => setIncludeVision(event.target.checked)}
              />
              이미지까지 전송
            </label>
            <label className="chat-toggle small">
              최대 이미지
              <input
                type="number"
                min="1"
                max="20"
                value={maxVisionImages}
                onChange={(event) => setMaxVisionImages(event.target.value)}
                disabled={!includeVision}
              />
            </label>
          </div>
        </div>

        <div className="chat-window">
          {chatMessages.length === 0 && (
            <p className="muted">아직 대화가 없습니다. 토픽을 입력해보세요.</p>
          )}
          {chatMessages.map((message, idx) => (
            <div key={`${message.role}-${idx}`} className={`chat-row ${message.role}`}>
              <div className={`chat-bubble ${message.role}`}>
                <div className="chat-meta">
                  {message.role === "user" ? "사용자" : "LLM"}
                </div>
                <p>{message.content}</p>
              </div>
            </div>
          ))}
          {chatLoading && (
            <div className="chat-row assistant">
              <div className="chat-bubble assistant">
                <div className="chat-meta">LLM</div>
                <p>응답을 생성 중입니다...</p>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {chatError && <div className="error">{chatError}</div>}

        <div className="chat-input">
          <textarea
            rows={3}
            placeholder="연구 토픽이나 질문을 입력하세요."
            value={chatPrompt}
            onChange={(event) => setChatPrompt(event.target.value)}
          />
          <button type="button" disabled={chatLoading} onClick={handleDiscuss}>
            {chatLoading ? "요청 중..." : "LLM에게 요청"}
          </button>
        </div>
      </section>

      <section className="grid">
        {grouped.length === 0 && (
          <div className="empty">아직 업로드된 이미지가 없습니다.</div>
        )}

        {grouped.map((group) => (
          <div key={group.key} className="card">
            <div className="card-header">
              <span className="tag">{group.label}</span>
              <span className="count">{group.items.length}장</span>
            </div>
            <div className="thumbs">
              {group.items.map((item) => {
                const isRevealed = isAuthed && revealedImages.includes(item.id);
                return (
                  <figure key={item.id} className={isRevealed ? "" : "locked"}>
                    {isRevealed ? (
                      <img
                        src={`${API_BASE}/images/${item.id}/file`}
                        alt={item.original_filename}
                        loading="lazy"
                      />
                    ) : (
                      <div className="thumb-placeholder">
                        <span>비밀번호 필요</span>
                        <button type="button" onClick={() => revealImage(item.id)}>
                          이미지 보기
                        </button>
                      </div>
                    )}
                    <figcaption>{item.original_filename}</figcaption>
                    <div className="meta">
                      <span>
                        페이지: {item.page_number ?? "미인식"}
                      </span>
                      <span>
                        클러스터: {item.cluster_id ?? "미분류"}
                      </span>
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
                        onChange={(event) =>
                          handleAssignBook(item.id, event.target.value)
                        }
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
                );
              })}
            </div>
          </div>
        ))}
      </section>

      {showAuthModal && (
        <div className="modal-backdrop" onClick={closeAuthModal}>
          <div className="modal" onClick={(event) => event.stopPropagation()}>
            <div>
              <h2>비밀번호 입력</h2>
              <p className="muted">
                {authReason || "사진을 보려면 비밀번호를 입력해야 합니다."}
              </p>
            </div>
            <form className="auth-form" onSubmit={handleAuthSubmit}>
              <input
                type="password"
                placeholder="비밀번호"
                value={passwordInput}
                onChange={(event) => setPasswordInput(event.target.value)}
              />
              <button type="submit">확인</button>
            </form>
            {authError && <div className="error">{authError}</div>}
            <div className="modal-actions">
              <button type="button" className="secondary" onClick={closeAuthModal}>
                취소
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
