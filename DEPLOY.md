# 배포 가이드

다른 사람이 이 RAG 에이전트를 사용할 수 있도록 배포하는 방법입니다.

## 방법 1: Streamlit Cloud 배포 (가장 쉬움) ⭐

### 장점
- 무료
- 자동 HTTPS
- GitHub 연동
- 쉬운 설정

### 단계

1. **GitHub에 프로젝트 업로드**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Streamlit Cloud 접속**
   - https://streamlit.io/cloud 접속
   - GitHub 계정으로 로그인

3. **앱 배포**
   - "New app" 클릭
   - Repository: 프로젝트 저장소 선택
   - Branch: `main` 또는 `master`
   - Main file path: `app.py`
   - App URL: 원하는 URL 입력 (예: `rag-agent`)

4. **Secrets 설정**
   - "Advanced settings" → "Secrets" 클릭
   - 다음 형식으로 입력:
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   TAVILY_API_KEY=tvly-your-actual-key-here
   ```

5. **Deploy!**
   - "Deploy" 버튼 클릭
   - 몇 분 후 배포 완료

### 공유하기
배포된 URL을 다른 사람에게 공유하면 됩니다!
예: `https://rag-agent.streamlit.app`

---

## 방법 2: 로컬 네트워크 공유

같은 네트워크(Wi-Fi 등)에 있는 사람들과 공유하는 방법입니다.

### 단계

1. **IP 주소 확인**
   ```bash
   # Windows
   ipconfig
   
   # macOS/Linux
   ifconfig
   ```
   IPv4 주소 확인 (예: `192.168.1.100`)

2. **Streamlit 실행**
   ```bash
   streamlit run app.py --server.address 0.0.0.0 --server.port 8501
   ```

3. **방화벽 설정**
   - Windows: 방화벽에서 Python 허용
   - macOS: 시스템 설정 → 보안 → 방화벽 설정

4. **공유**
   - 다른 사람에게 `http://<your-ip>:8501` 주소 공유
   - 예: `http://192.168.1.100:8501`

---

## 방법 3: GitHub 공유

코드를 GitHub에 올려서 다른 사람이 클론해서 사용하도록 합니다.

### 단계

1. **GitHub 저장소 생성**
   - https://github.com/new 접속
   - 저장소 이름 입력
   - Public 또는 Private 선택
   - 생성

2. **프로젝트 업로드**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<username>/<repo-name>.git
   git push -u origin main
   ```

3. **README 작성**
   - README.md에 설치 및 사용 방법 명시
   - API 키 발급 방법 안내

4. **공유**
   - GitHub 저장소 URL 공유
   - 다른 사람은 클론 후 `.env` 파일 설정하여 사용

---

## 방법 4: Docker 배포 (고급)

컨테이너화하여 배포하는 방법입니다.

### Dockerfile 생성

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

### 실행

```bash
docker build -t rag-agent .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key -e TAVILY_API_KEY=your_key rag-agent
```

---

## 보안 주의사항 ⚠️

1. **API 키 보호**
   - `.env` 파일을 절대 GitHub에 커밋하지 마세요
   - `.gitignore`에 `.env`가 포함되어 있는지 확인

2. **공개 배포 시**
   - API 키를 코드에 하드코딩하지 마세요
   - Streamlit Secrets 또는 환경 변수 사용

3. **비용 관리**
   - OpenAI API 사용량 모니터링
   - 사용량 제한 설정 고려

---

## 추천 배포 방법

- **개인 프로젝트/테스트**: 방법 2 (로컬 네트워크)
- **공개 배포**: 방법 1 (Streamlit Cloud) ⭐
- **기업/프라이빗**: 방법 3 (GitHub Private) 또는 방법 4 (Docker)

