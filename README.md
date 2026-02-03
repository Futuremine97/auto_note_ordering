# auto_note_ordering

Page OCR Sorter

책 스캔 이미지를 업로드하면 OCR로 페이지 번호를 인식해 자동으로 분류하는 웹앱입니다.

## 구성
- `backend/`: FastAPI + PostgreSQL + Tesseract
- `frontend/`: React (Vite)
- `docker-compose.yml`: PostgreSQL

## 실행 방법 (로컬 개발)
1. PostgreSQL 실행
```bash
docker compose up -d
```

2. 백엔드 실행
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

3. 프론트엔드 실행
```bash
cd frontend
npm install
npm run dev
```

기본 API 주소는 `http://localhost:8000`입니다.

## 공개 배포 (Nginx + HTTPS + 도메인)
도메인: `your-domain.com`

### 1) DNS 설정
도메인 관리 페이지에서 다음 A 레코드를 추가하세요.
- `@` → 서버 공인 IP
- `www` → 서버 공인 IP

### 2) 서버 준비 (Ubuntu VPS 가정)
```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo systemctl enable --now docker
```

### 3) 배포 실행
서버에 이 프로젝트를 클론한 뒤:
```bash
docker compose -f docker-compose.prod.yml up -d --build
```

### 4) HTTPS 설정 (Let’s Encrypt)
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

인증서 자동 갱신 확인:
```bash
sudo certbot renew --dry-run
```

### 5) 방화벽 (선택)
```bash
sudo ufw allow OpenSSH
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

## Tesseract 설치
- macOS (Homebrew)
```bash
brew install tesseract
```

Tesseract 경로가 PATH에 없으면 `backend/.env`에 `TESSERACT_CMD`를 지정하세요.
한국어 페이지 번호 인식이 필요하면 `TESSERACT_LANG=eng+kor`로 설정하고 언어 팩을 설치하세요.

## OCR 정확도 팁
페이지 번호는 상단/하단 밴드를 별도 OCR로 재검사하고, 상단을 약간 더 선호하도록
설정했습니다. 스캔 품질이 낮다면 해상도를 높이고, 페이지 번호가 잘 보이도록
크롭해서 업로드하면 정확도가 올라갑니다.

## 환경 변수
`backend/.env.example` 참고

### 프로덕션용 환경 변수 (.env)
배포 서버에서 아래 예시를 참고해 `.env`를 만들고 **강력한 비밀번호로 변경**하세요.
```bash
cp .env.example .env
```

## 책별 저자 분류 (n-gram)
OCR 텍스트를 책별로 라벨링한 뒤 n-gram 모델로 분류합니다.

### 흐름
1. 책 등록
2. 이미지에 책 ID 라벨 지정
3. 책별 모델 학습
4. 이미지별 예측 실행

### API 예시
```bash
# 1) 책 등록
curl -X POST http://localhost:8000/api/books \
  -H "Content-Type: application/json" \
  -d '{"title":"책 제목","author_name":"저자 이름"}'

# 2) 이미지에 책 지정
curl -X PATCH http://localhost:8000/api/images/1 \
  -H "Content-Type: application/json" \
  -d '{"book_id":1}'

# 3) 모델 학습
curl -X POST http://localhost:8000/api/books/1/train

# 4) 예측
curl -X POST http://localhost:8000/api/images/1/predict
```

### DB 주의
기존 DB를 사용 중이라면 새 컬럼 추가가 필요합니다. 간단히 시작하려면
로컬/개발 DB를 삭제 후 재생성하거나, 마이그레이션을 적용하세요.
