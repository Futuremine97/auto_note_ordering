# auto_note_ordering

Page OCR Sorter

A web app that uploads scanned book images, detects page numbers via OCR, and automatically organizes pages. It also supports book/author classification with n-gram models, clustering, and LLM discussion (OCR + optional images).

## Structure
- `backend/`: FastAPI + PostgreSQL + Tesseract
- `frontend/`: React (Vite)
- `docker-compose.yml`: PostgreSQL (local)
- `docker-compose.prod.yml`: Full stack (backend + db + nginx)

## Local Development
1) Start PostgreSQL
```bash
docker compose up -d
```

2) Run backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

3) Run frontend
```bash
cd frontend
npm install
npm run dev
```

Default API base: `http://localhost:8000`

## Production (Nginx + HTTPS + Domain)
Domain: `your-domain.com`

### 1) DNS
Create A records in your domain DNS:
- `@` → server public IP
- `www` → server public IP

### 2) Server setup (Ubuntu VPS)
```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo systemctl enable --now docker
```

### 3) Deploy
```bash
docker compose -f docker-compose.prod.yml up -d --build
```

### 4) HTTPS (Let’s Encrypt)
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

Renew test:
```bash
sudo certbot renew --dry-run
```

### 5) Firewall (optional)
```bash
sudo ufw allow OpenSSH
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

## Tesseract
- macOS (Homebrew)
```bash
brew install tesseract
```

If Tesseract is not in PATH, set `TESSERACT_CMD` in `.env`.
For Korean/Japanese page numbers set `TESSERACT_LANG=eng+kor+jpn` and install language packs.

## OCR Accuracy Tips
We scan top/bottom bands and slightly prefer the top area. For better accuracy:
- increase scan resolution
- avoid cropping out the page number
- keep numbers near the top/bottom margin

## Environment Variables
See `./.env.example`

### Photo Access Password (optional)
Set a password to require login before viewing photos/OCR data.
```bash
PHOTO_PASSWORD=your_password
AUTH_SECRET=long_random_secret
AUTH_COOKIE_SECURE=true
AUTH_COOKIE_TTL_HOURS=72
```

If you run without HTTPS (local dev), set `AUTH_COOKIE_SECURE=false`.

### Production `.env`
Create `.env` on the server and use strong passwords:
```bash
cp .env.example .env
```

## Book / Author Classification (n-gram)
Label images by book, train n-gram models, then predict.

### Accumulated vocab
The global n-gram vocab is accumulated in DB and used for stable smoothing.

### Perplexity tuning
```bash
curl -X POST http://localhost:8000/api/ngram/tune \
  -H "Content-Type: application/json" \
  -d '{
    "n_values_options": [[3,4,5],[2,3,4,5],[3,4]],
    "alphas": [0.5, 1.0, 1.5],
    "train_ratio": 0.8,
    "random_seed": 42
  }'
```

### Flow
1) Create book
2) Assign book ID to images
3) Train book model
4) Predict per image / bulk

### API Examples
```bash
# 1) create book
curl -X POST http://localhost:8000/api/books \
  -H "Content-Type: application/json" \
  -d '{"title":"Book Title","author_name":"Author Name"}'

# 2) assign book
curl -X PATCH http://localhost:8000/api/images/1 \
  -H "Content-Type: application/json" \
  -d '{"book_id":1}'

# 3) train
curl -X POST http://localhost:8000/api/books/1/train

# 4) predict
curl -X POST http://localhost:8000/api/images/1/predict
```

## Clustering
Cluster similar OCR pages and optionally apply labels.

```bash
curl -X POST http://localhost:8000/api/images/cluster \
  -H "Content-Type: application/json" \
  -d '{"threshold":0.25,"apply_labels":false}'
```

Max clusters are fixed to 10.

## LLM Discussion (OCR + optional images)
The UI can send OCR text (and optionally images) to an LLM for research discussion.

### OpenAI-compatible example
```bash
LLM_PROVIDER=openai_compatible
LLM_API_BASE=https://api.openai.com
LLM_API_KEY=YOUR_KEY
LLM_MODEL=YOUR_MODEL
```

## DB Notes
If you use an existing DB, you may need to add new columns (e.g., `book_id`, `predicted_*`, `cluster_id`).
For a clean start, reset the DB or run migrations/ALTER TABLE commands.
