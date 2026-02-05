# Auto Note Ordering (Page OCR Sorter)

Upload scanned book pages, detect page numbers via OCR, and automatically organize pages by book or cluster. The app also supports n‑gram classification and optional LLM discussions based on OCR text (and images if enabled).

## Features
- OCR page number detection (top/bottom bands with top bias)
- Automatic grouping by book, prediction, or cluster
- n‑gram training, perplexity tuning, and bulk prediction
- Optional LLM discussion UI (OCR + optional images)
- Password gate for viewing photos/OCR

## Tech Stack
- Backend: FastAPI + PostgreSQL + Tesseract
- Frontend: React (Vite)
- Infra: Docker Compose + Nginx + HTTPS

## Project Layout
- `backend/`: FastAPI app
- `frontend/`: React app
- `docker-compose.yml`: local DB
- `docker-compose.prod.yml`: full stack

## Local Development
1. Start Postgres
```bash
docker compose up -d
```

2. Run backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

3. Run frontend
```bash
cd frontend
npm install
npm run dev
```

Default API base: `http://localhost:8000`

## Production (Nginx + HTTPS)
### 1) DNS
Create A records:
- `@` → server public IP
- `www` → server public IP

### 2) Server setup
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

## OCR Notes
If Tesseract is not in PATH, set `TESSERACT_CMD` in `.env`.
Recommended languages: `TESSERACT_LANG=eng+kor+jpn`.

## Environment Variables
See `./.env.example`.

### Password Gate (recommended for public repos)
Set a password to require login before viewing photos/OCR:
```bash
PHOTO_PASSWORD=your_password
AUTH_SECRET=long_random_secret
AUTH_COOKIE_SECURE=true
AUTH_COOKIE_TTL_HOURS=72
```

If you run without HTTPS (local dev), set `AUTH_COOKIE_SECURE=false`.

## Book / Author Classification (n‑gram)
Flow:
1. Create book
2. Assign book ID to images
3. Train book model
4. Predict per image or bulk

Perplexity tuning:
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

## Clustering
```bash
curl -X POST http://localhost:8000/api/images/cluster \
  -H "Content-Type: application/json" \
  -d '{"threshold":0.25,"apply_labels":false}'
```

Max clusters are fixed to 10.

## LLM Discussion (optional)
To disable costs, set:
```bash
LLM_PROVIDER=mock
```

OpenAI-compatible example:
```bash
LLM_PROVIDER=openai_compatible
LLM_API_BASE=https://api.openai.com
LLM_API_KEY=YOUR_KEY
LLM_MODEL=YOUR_MODEL
```

## Security Notes
- Do not commit `.env` or API keys to a public repo.
- Revoke any tokens that were pasted into chats or logs.
