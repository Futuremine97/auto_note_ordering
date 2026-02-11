import { useCallback, useEffect, useMemo, useRef, useState } from "react";

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

function clusterColor(key) {
  if (key === "미분류") return "rgba(128, 128, 128, 0.6)";
  const id = Number(key);
  const hue = (id * 57) % 360;
  return `hsl(${hue} 70% 55%)`;
}

function useResizeObserver(ref, onResize) {
  useEffect(() => {
    if (!ref.current || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        onResize(entry.contentRect);
      }
    });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [ref, onResize]);
}

function Cluster3D({ points, outlierIds = new Set(), height = 360 }) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const projectedRef = useRef([]);
  const [size, setSize] = useState({ width: 0, height });
  const [rotation, setRotation] = useState({ x: 0.6, y: 0.8 });
  const [zoom, setZoom] = useState(1);
  const [hovered, setHovered] = useState(null);

  useResizeObserver(containerRef, (rect) => {
    setSize({ width: rect.width, height });
  });

  const normalized = useMemo(() => {
    if (!points.length) return [];
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const zs = points.map((p) => p.z);
    const center = {
      x: xs.reduce((a, b) => a + b, 0) / xs.length,
      y: ys.reduce((a, b) => a + b, 0) / ys.length,
      z: zs.reduce((a, b) => a + b, 0) / zs.length,
    };
    const maxRange = Math.max(
      1,
      ...xs.map((value) => Math.abs(value - center.x)),
      ...ys.map((value) => Math.abs(value - center.y)),
      ...zs.map((value) => Math.abs(value - center.z))
    );
    return points.map((point) => ({
      ...point,
      nx: (point.x - center.x) / maxRange,
      ny: (point.y - center.y) / maxRange,
      nz: (point.z - center.z) / maxRange,
    }));
  }, [points]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.width === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const centerX = size.width / 2;
    const centerY = size.height / 2;
    const scale = Math.min(size.width, size.height) * 0.42 * zoom;
    const cosX = Math.cos(rotation.x);
    const sinX = Math.sin(rotation.x);
    const cosY = Math.cos(rotation.y);
    const sinY = Math.sin(rotation.y);
    const fov = 2.6;

    ctx.clearRect(0, 0, size.width, size.height);

    const projectPoint = (nx, ny, nz) => {
      const xz = nx * cosY + nz * sinY;
      const zz = -nx * sinY + nz * cosY;
      const yz = ny * cosX - zz * sinX;
      const zz2 = ny * sinX + zz * cosX;
      const depth = fov + zz2;
      const perspective = scale / depth;
      return {
        sx: centerX + xz * perspective,
        sy: centerY - yz * perspective,
        depth,
        r: Math.max(2, 3.4 / depth + 1.2),
      };
    };

    const projected = normalized.map((point) => {
      const projectedPoint = projectPoint(point.nx, point.ny, point.nz);
      return {
        ...point,
        ...projectedPoint,
        isOutlier: outlierIds.has(point.id),
      };
    });

    projected.sort((a, b) => a.depth - b.depth);
    projectedRef.current = projected;

    const origin = projectPoint(0, 0, 0);
    const axisX = projectPoint(1, 0, 0);
    const axisY = projectPoint(0, 1, 0);
    const axisZ = projectPoint(0, 0, 1);

    const drawAxis = (target, color, label) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.2;
      ctx.moveTo(origin.sx, origin.sy);
      ctx.lineTo(target.sx, target.sy);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.font = "12px 'Pretendard', sans-serif";
      ctx.fillText(label, target.sx + 4, target.sy - 4);
    };

    const drawGridLine = (start, end) => {
      const a = projectPoint(start[0], start[1], start[2]);
      const b = projectPoint(end[0], end[1], end[2]);
      ctx.beginPath();
      ctx.strokeStyle = "rgba(120, 120, 140, 0.25)";
      ctx.lineWidth = 1;
      ctx.moveTo(a.sx, a.sy);
      ctx.lineTo(b.sx, b.sy);
      ctx.stroke();
    };

    const gridTicks = [-1, -0.5, 0, 0.5, 1];
    // Grid on three main planes (back, left, bottom)
    for (const t of gridTicks) {
      drawGridLine([-1, t, -1], [1, t, -1]);
      drawGridLine([t, -1, -1], [t, 1, -1]);

      drawGridLine([-1, -1, t], [1, -1, t]);
      drawGridLine([t, -1, -1], [t, -1, 1]);

      drawGridLine([-1, t, -1], [-1, t, 1]);
      drawGridLine([-1, -1, t], [-1, 1, t]);
    }

    drawAxis(axisX, "rgba(232, 93, 93, 0.9)", "Principal Component 1");
    drawAxis(axisY, "rgba(88, 181, 110, 0.9)", "Principal Component 2");
    drawAxis(axisZ, "rgba(82, 131, 255, 0.9)", "Principal Component 3");

    const drawStar = (x, y, radius) => {
      const spikes = 5;
      const step = Math.PI / spikes;
      let rot = (Math.PI / 2) * 3;
      ctx.beginPath();
      ctx.moveTo(x, y - radius);
      for (let i = 0; i < spikes; i += 1) {
        ctx.lineTo(x + Math.cos(rot) * radius, y + Math.sin(rot) * radius);
        rot += step;
        ctx.lineTo(
          x + Math.cos(rot) * (radius * 0.45),
          y + Math.sin(rot) * (radius * 0.45)
        );
        rot += step;
      }
      ctx.lineTo(x, y - radius);
      ctx.closePath();
      ctx.fill();
    };

    for (const point of projected) {
      ctx.globalAlpha = 0.9;
      if (point.isOutlier) {
        ctx.fillStyle = "rgba(239, 68, 68, 0.9)";
        drawStar(point.sx, point.sy, point.r * 1.35);
      } else {
        ctx.beginPath();
        ctx.fillStyle = "rgba(59, 130, 246, 0.75)";
        ctx.arc(point.sx, point.sy, point.r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }, [normalized, rotation, zoom, size, outlierIds]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    const handleDown = (event) => {
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
    };
    const handleMove = (event) => {
      if (!dragging) return;
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      setRotation((prev) => ({
        x: prev.x + dy * 0.005,
        y: prev.y + dx * 0.005,
      }));
    };
    const handleUp = () => {
      dragging = false;
    };
    const handleWheel = (event) => {
      event.preventDefault();
      setZoom((prev) => Math.min(3, Math.max(0.5, prev - event.deltaY * 0.001)));
    };

    const handleHover = (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      let closest = null;
      let minDist = Infinity;
      for (const point of projectedRef.current) {
        const dx = point.sx - x;
        const dy = point.sy - y;
        const dist = dx * dx + dy * dy;
        const radius = (point.r || 3) + 6;
        if (dist <= radius * radius && dist < minDist) {
          minDist = dist;
          closest = point;
        }
      }
      if (closest) {
        setHovered({ ...closest, px: x, py: y });
      } else {
        setHovered(null);
      }
    };

    const handleLeave = () => {
      setHovered(null);
    };

    canvas.addEventListener("mousedown", handleDown);
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    canvas.addEventListener("mousemove", handleHover);
    canvas.addEventListener("mouseleave", handleLeave);

    return () => {
      canvas.removeEventListener("mousedown", handleDown);
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
      canvas.removeEventListener("wheel", handleWheel);
      canvas.removeEventListener("mousemove", handleHover);
      canvas.removeEventListener("mouseleave", handleLeave);
    };
  }, []);

  return (
    <div className="cluster-canvas-wrap" ref={containerRef} style={{ height }}>
      <canvas
        ref={canvasRef}
        width={size.width}
        height={size.height}
        className="cluster-canvas"
      />
      {hovered && (
        <div
          className="cluster-tooltip"
          style={{ left: hovered.px, top: hovered.py }}
        >
          <img
            src={`${API_BASE}/images/${hovered.id}/file`}
            alt={`image-${hovered.id}`}
            loading="lazy"
          />
          <div className="cluster-tooltip-meta">
            <strong>이미지 #{hovered.id}</strong>
            <span>페이지: {hovered.page_number ?? "미인식"}</span>
          </div>
        </div>
      )}
    </div>
  );
}

function Cluster2D({ points, outlierIds = new Set(), height = 320 }) {
  const containerRef = useRef(null);
  const [size, setSize] = useState({ width: 0, height });

  useResizeObserver(containerRef, (rect) => {
    setSize({ width: rect.width, height });
  });

  const normalized = useMemo(() => {
    if (!points.length) return [];
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const center = {
      x: xs.reduce((a, b) => a + b, 0) / xs.length,
      y: ys.reduce((a, b) => a + b, 0) / ys.length,
    };
    const maxRange = Math.max(
      1,
      ...xs.map((value) => Math.abs(value - center.x)),
      ...ys.map((value) => Math.abs(value - center.y))
    );
    return points.map((point) => ({
      ...point,
      nx: (point.x - center.x) / maxRange,
      ny: (point.y - center.y) / maxRange,
    }));
  }, [points]);

  const padding = 42;
  const width = Math.max(size.width, 1);
  const heightPx = height;
  const ticks = [-1, -0.5, 0, 0.5, 1];

  const toScreen = (nx, ny) => {
    const x = padding + ((nx + 1) / 2) * (width - padding * 2);
    const y = padding + (1 - (ny + 1) / 2) * (heightPx - padding * 2);
    return { x, y };
  };

  const starPath = (cx, cy, radius) => {
    const spikes = 5;
    const step = Math.PI / spikes;
    let rot = (Math.PI / 2) * 3;
    let path = `M ${cx} ${cy - radius}`;
    for (let i = 0; i < spikes; i += 1) {
      path += ` L ${cx + Math.cos(rot) * radius} ${cy + Math.sin(rot) * radius}`;
      rot += step;
      path += ` L ${cx + Math.cos(rot) * radius * 0.45} ${cy + Math.sin(rot) * radius * 0.45}`;
      rot += step;
    }
    path += " Z";
    return path;
  };

  return (
    <div className="cluster-2d-wrap" ref={containerRef} style={{ height }}>
      <svg
        className="cluster-2d"
        width={width}
        height={heightPx}
        viewBox={`0 0 ${width} ${heightPx}`}
      >
        <rect
          x={0}
          y={0}
          width={width}
          height={heightPx}
          fill="transparent"
        />
        {ticks.map((t) => {
          const x = padding + ((t + 1) / 2) * (width - padding * 2);
          const y = padding + (1 - (t + 1) / 2) * (heightPx - padding * 2);
          return (
            <g key={`grid-${t}`}>
              <line
                x1={x}
                y1={padding}
                x2={x}
                y2={heightPx - padding}
                stroke="rgba(120,120,140,0.25)"
                strokeWidth="1"
              />
              <line
                x1={padding}
                y1={y}
                x2={width - padding}
                y2={y}
                stroke="rgba(120,120,140,0.25)"
                strokeWidth="1"
              />
            </g>
          );
        })}
        <line
          x1={padding}
          y1={heightPx - padding}
          x2={width - padding}
          y2={heightPx - padding}
          stroke="rgba(232,93,93,0.9)"
          strokeWidth="1.2"
        />
        <line
          x1={padding}
          y1={padding}
          x2={padding}
          y2={heightPx - padding}
          stroke="rgba(88,181,110,0.9)"
          strokeWidth="1.2"
        />
        <text x={width - padding + 8} y={heightPx - padding + 4} className="cluster-2d-axis">
          PC1
        </text>
        <text x={padding - 28} y={padding - 8} className="cluster-2d-axis">
          PC2
        </text>
        {normalized.map((point) => {
          const { x, y } = toScreen(point.nx, point.ny);
          if (outlierIds.has(point.id)) {
            return (
              <path
                key={point.id}
                d={starPath(x, y, 6)}
                fill="rgba(239, 68, 68, 0.9)"
              />
            );
          }
          return (
            <circle
              key={point.id}
              cx={x}
              cy={y}
              r={3.2}
              fill="rgba(59, 130, 246, 0.75)"
            />
          );
        })}
      </svg>
    </div>
  );
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
  const [embeddingL1, setEmbeddingL1] = useState({ points3d: [], points2d: [] });
  const [embeddingL2, setEmbeddingL2] = useState({ points3d: [], points2d: [] });
  const [embeddingLoading, setEmbeddingLoading] = useState(false);
  const [embeddingError, setEmbeddingError] = useState("");
  const [embeddingStatus, setEmbeddingStatus] = useState("");
  const [embeddingStatusMessage, setEmbeddingStatusMessage] = useState("");
  const embeddingRetryRef = useRef(null);
  const [audioText, setAudioText] = useState("");
  const [audioLoading, setAudioLoading] = useState(false);
  const [audioError, setAudioError] = useState("");
  const [audioFileName, setAudioFileName] = useState("");
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
  const embeddingPoints = embeddingL2.points3d;
  const isEmbeddingProcessing =
    embeddingLoading || embeddingStatus === "processing";

  const grouped = useMemo(() => {
    return viewMode === "cluster"
      ? groupByCluster(records)
      : groupByBook(records, books);
  }, [records, books, viewMode]);

  const clusterStats = useMemo(() => {
    const counts = new Map();
    for (const record of records) {
      const key = record.cluster_id ?? "미분류";
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    const list = [...counts.entries()].map(([key, count]) => ({
      key,
      count,
      label: key === "미분류" ? "미분류" : `클러스터 ${key}`,
    }));
    list.sort((a, b) => {
      if (a.key === "미분류") return 1;
      if (b.key === "미분류") return -1;
      return Number(a.key) - Number(b.key);
    });
    const max = Math.max(1, ...list.map((item) => item.count));
    return { list, max };
  }, [records]);

  const buildOutlierSet = useCallback((points) => {
    if (!points || points.length === 0) return new Set();
    const counts = new Map();
    for (const point of points) {
      const key = point.cluster_id ?? "미분류";
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    const outliers = new Set();
    for (const point of points) {
      const key = point.cluster_id ?? "미분류";
      const size = counts.get(key) || 0;
      if (point.cluster_id == null || size <= 1) {
        outliers.add(point.id);
      }
    }
    return outliers;
  }, []);

  const l1Outliers = useMemo(
    () => buildOutlierSet(embeddingL1.points3d),
    [embeddingL1.points3d, buildOutlierSet]
  );

  const l2Outliers = useMemo(
    () => buildOutlierSet(embeddingL2.points3d),
    [embeddingL2.points3d, buildOutlierSet]
  );

  const clusterCenterIds = useMemo(() => {
    if (embeddingPoints.length === 0) return new Set();
    const byCluster = new Map();
    for (const point of embeddingPoints) {
      if (point.cluster_id == null) continue;
      if (!byCluster.has(point.cluster_id)) {
        byCluster.set(point.cluster_id, []);
      }
      byCluster.get(point.cluster_id).push(point);
    }
    const centers = new Set();
    for (const [, points] of byCluster.entries()) {
      if (points.length < 2) continue;
      const cx = points.reduce((sum, p) => sum + p.x, 0) / points.length;
      const cy = points.reduce((sum, p) => sum + p.y, 0) / points.length;
      const cz = points.reduce((sum, p) => sum + p.z, 0) / points.length;
      let best = null;
      let bestDist = Infinity;
      for (const point of points) {
        const dx = point.x - cx;
        const dy = point.y - cy;
        const dz = point.z - cz;
        const dist = dx * dx + dy * dy + dz * dz;
        if (dist < bestDist) {
          bestDist = dist;
          best = point;
        }
      }
      if (best) centers.add(best.id);
    }
    return centers;
  }, [embeddingPoints]);

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

  async function fetchEmbedding() {
    if (!isAuthed) {
      setEmbeddingL1({ points3d: [], points2d: [] });
      setEmbeddingL2({ points3d: [], points2d: [] });
      setEmbeddingStatus("");
      setEmbeddingStatusMessage("");
      return;
    }
    if (records.length === 0) {
      setEmbeddingL1({ points3d: [], points2d: [] });
      setEmbeddingL2({ points3d: [], points2d: [] });
      setEmbeddingError("");
      setEmbeddingStatus("");
      setEmbeddingStatusMessage("");
      return;
    }
    setEmbeddingError("");
    setEmbeddingStatus("");
    setEmbeddingStatusMessage("");
    setEmbeddingLoading(true);
    try {
      const res = await fetch(`${API_BASE}/images/cluster-embedding-compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (res.status === 202) {
        const data = await res.json();
        setEmbeddingStatus("processing");
        setEmbeddingStatusMessage(
          data?.detail ||
            "임베딩 계산 중입니다. 첫 호출은 수 분이 걸릴 수 있습니다."
        );
        if (embeddingRetryRef.current) {
          clearTimeout(embeddingRetryRef.current);
        }
        embeddingRetryRef.current = setTimeout(() => {
          embeddingRetryRef.current = null;
          fetchEmbedding();
        }, 5000);
        return;
      }
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "3D 시각화 데이터를 불러오지 못했습니다.");
      }
      const data = await res.json();
      setEmbeddingL1({
        points3d: data.l1?.points_3d || [],
        points2d: data.l1?.points_2d || [],
      });
      setEmbeddingL2({
        points3d: data.l2?.points_3d || [],
        points2d: data.l2?.points_2d || [],
      });
      setEmbeddingStatus("ready");
      setEmbeddingStatusMessage("");
    } catch (err) {
      setEmbeddingError(err.message);
      setEmbeddingStatus("");
      setEmbeddingStatusMessage("");
    } finally {
      setEmbeddingLoading(false);
    }
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
    if (!authChecked || !isAuthed) return;
    fetchEmbedding();
  }, [authChecked, isAuthed, records.length]);

  useEffect(() => {
    return () => {
      if (embeddingRetryRef.current) {
        clearTimeout(embeddingRetryRef.current);
      }
    };
  }, []);

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
    await fetchEmbedding();
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

  async function handleAudioUpload(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    if (!requestAuth("음성 메모 업로드를 하려면 비밀번호가 필요합니다.")) {
      event.target.value = "";
      return;
    }
    const file = files[0];
    setAudioFileName(file.name);
    setAudioText("");
    setAudioError("");
    setAudioLoading(true);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/audio/transcribe`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const message = await res.text();
        throw new Error(message || "Transcription failed");
      }
      const data = await res.json();
      setAudioText(data.text || "");
    } catch (err) {
      setAudioError(err.message || "Transcription failed");
    } finally {
      setAudioLoading(false);
      event.target.value = "";
    }
  }

  function handleAudioDownload() {
    if (!audioText) return;
    const blob = new Blob([audioText], { type: "text/plain;charset=utf-8" });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    const baseName = audioFileName
      ? audioFileName.replace(/\.[^.]+$/, "")
      : "transcript";
    link.href = url;
    link.download = `${baseName}.txt`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
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

      <section className="panel cluster-panel">
        <div>
          <h2>클러스터 분포</h2>
          <p className="muted">
            현재 업로드된 이미지가 어떤 클러스터에 얼마나 분포되어 있는지 시각화합니다.
          </p>
        </div>
        {clusterStats.list.length === 0 ? (
          <p className="muted">
            아직 클러스터링 결과가 없습니다. "이미지 클러스터링"을 실행하면 분포가 표시됩니다.
          </p>
        ) : (
          <div className="cluster-bars">
            {clusterStats.list.map((item) => (
              <div key={item.key} className="cluster-bar-row">
                <div className="cluster-bar-label">{item.label}</div>
                <div className="cluster-bar-track">
                  <div
                    className="cluster-bar-fill"
                    style={{
                      width: `${(item.count / clusterStats.max) * 100}%`,
                      background: clusterColor(item.key),
                    }}
                  />
                </div>
                <div className="cluster-bar-count">{item.count}장</div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className={`panel cluster-embedding-panel ${!isAuthed ? "disabled-panel" : ""}`}>
        <div className="cluster-embedding-header">
          <div>
            <h2>사진 3D 시각화</h2>
            <p className="muted">
              SOTA 임베딩 + 오토인코더로 학습한 L1/L2 representation을 3D/2D 산점도로 확인합니다.
            </p>
            <p className="muted">
              점 위에 마우스를 올리면 해당 사진이 미리보기로 표시됩니다.
            </p>
          </div>
          <div className="cluster-embedding-actions">
            <button
              type="button"
              onClick={fetchEmbedding}
              disabled={!isAuthed || isEmbeddingProcessing}
            >
              {isEmbeddingProcessing ? "계산 중..." : "임베딩 새로고침"}
            </button>
          </div>
        </div>
        {embeddingStatus === "processing" && embeddingStatusMessage && (
          <p className="muted">{embeddingStatusMessage}</p>
        )}
        {!isAuthed ? (
          <p className="muted">비밀번호를 입력하면 3D 시각화를 확인할 수 있습니다.</p>
        ) : embeddingError ? (
          <div className="error">{embeddingError}</div>
        ) : embeddingStatus === "processing" ? (
          <p className="muted">임베딩을 계산 중입니다. 완료되면 자동으로 갱신됩니다.</p>
        ) : embeddingL1.points3d.length === 0 && embeddingL2.points3d.length === 0 ? (
          <p className="muted">
            아직 시각화 데이터가 없습니다. 이미지 업로드 또는 클러스터링 이후 확인하세요.
          </p>
        ) : (
          <>
            <div className="embedding-views">
              <div className="embedding-view">
                <h3>L1 Loss · 3D Scatter</h3>
                <Cluster3D points={embeddingL1.points3d} outlierIds={l1Outliers} />
              </div>
              <div className="embedding-view">
                <h3>L1 Loss · 2D Scatter</h3>
                <Cluster2D points={embeddingL1.points2d} outlierIds={l1Outliers} />
              </div>
              <div className="embedding-view">
                <h3>L2 Loss · 3D Scatter</h3>
                <Cluster3D points={embeddingL2.points3d} outlierIds={l2Outliers} />
              </div>
              <div className="embedding-view">
                <h3>L2 Loss · 2D Scatter</h3>
                <Cluster2D points={embeddingL2.points2d} outlierIds={l2Outliers} />
              </div>
            </div>
            <div className="embedding-legend">
              <div className="embedding-legend-item">
                <span className="embedding-legend-dot" />
                normal
              </div>
              <div className="embedding-legend-item">
                <span className="embedding-legend-star">★</span>
                outlier (클러스터 단독/미분류)
              </div>
            </div>
          </>
        )}
      </section>

      <section className={`panel audio-panel ${!isAuthed ? "disabled-panel" : ""}`}>
        <div className="audio-header">
          <div>
            <h2>Voice Memo → Text (English)</h2>
            <p className="muted">
              Upload an audio memo and get an English transcription.
            </p>
          </div>
          <label className={`upload audio-upload ${!isAuthed ? "disabled" : ""}`}>
            <input
              type="file"
              accept="audio/*"
              onChange={handleAudioUpload}
              disabled={!isAuthed || audioLoading}
            />
            {audioLoading ? "Transcribing..." : "오디오 업로드"}
          </label>
        </div>
        {audioText && (
          <div className="audio-actions">
            <button type="button" onClick={handleAudioDownload}>
              텍스트 다운로드
            </button>
          </div>
        )}
        {audioFileName && (
          <p className="muted">File: {audioFileName}</p>
        )}
        {audioError && <div className="error">{audioError}</div>}
        {audioText && (
          <pre className="audio-output">{audioText}</pre>
        )}
        {!audioText && !audioError && !audioLoading && (
          <p className="muted">
            Supports common formats (m4a, mp3, wav). Language is fixed to English.
          </p>
        )}
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
                const isCenter = clusterCenterIds.has(item.id);
                const figureClass = [
                  isRevealed ? "" : "locked",
                  isCenter ? "cluster-center" : "",
                ]
                  .filter(Boolean)
                  .join(" ");
                return (
                  <figure key={item.id} className={figureClass}>
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
