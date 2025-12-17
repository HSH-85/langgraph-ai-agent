# 🤖 LangGraph RAG 에이전트

LangGraph를 기반으로 한 RAG(Retrieval-Augmented Generation) 에이전트입니다. 벡터 스토어 검색과 웹 검색을 결합하여 정확한 답변을 생성합니다.

## ✨ 주요 기능

- **🔍 벡터 스토어 검색**: ChromaDB를 사용한 문서 검색
- **🌐 웹 검색**: Tavily를 통한 실시간 웹 검색
- **📝 답변 생성**: OpenAI GPT 모델을 사용한 답변 생성
- **💬 대화형 UI**: Streamlit 기반 사용자 인터페이스
- **🧠 지능형 라우팅**: 문서 관련성 평가 후 웹 검색 여부 자동 결정

## 📁 프로젝트 구조

```
aiAgent/
├── main.py          # 그래프 정의 및 실행
├── nodes.py         # 노드 함수들 (retrieve, grade_documents, web_search_node, generate)
├── state.py         # 상태 정의 (GraphState)
├── app.py           # Streamlit UI
├── requirements.txt # 의존성 목록
├── check_api_keys.py # API 키 확인 스크립트
└── README.md        # 프로젝트 설명
```

## 🚀 빠른 시작

### 1. 저장소 클론 또는 다운로드

```bash
git clone https://github.com/HSH-85/langgraph-ai-agent.git
cd langgraph-ai-agent
```

또는 ZIP 파일을 다운로드하여 압축 해제

### 2. 가상 환경 생성 및 활성화

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 다음 내용을 추가하세요:

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**API 키 발급:**
- OpenAI API 키: https://platform.openai.com/api-keys
- Tavily API 키: https://tavily.com (웹 검색용, 선택사항)

### 5. API 키 확인 (선택사항)

```bash
python check_api_keys.py
```

### 6. 실행

**Streamlit UI 실행 (권장):**
```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하세요.

**Python 스크립트로 실행:**
```bash
python main.py
```

## 📖 사용 방법

### Streamlit UI 사용

1. `streamlit run app.py` 실행
2. 브라우저에서 질문 입력
3. 에이전트가 자동으로:
   - 벡터 스토어에서 문서 검색
   - 문서 관련성 평가
   - 필요시 웹 검색 수행
   - 최종 답변 생성

### 문서 추가하기

PDF 문서를 벡터 스토어에 추가하려면:

```python
from nodes import load_pdf_to_vectorstore

# PDF 파일을 벡터 스토어에 추가
retriever = load_pdf_to_vectorstore("path/to/document.pdf")
```

## 🔧 환경 변수

| 변수명 | 설명 | 필수 여부 |
|--------|------|----------|
| `OPENAI_API_KEY` | OpenAI API 키 | ✅ 필수 |
| `TAVILY_API_KEY` | Tavily API 키 (웹 검색용) | ⚠️ 선택 |

## 🌐 배포 방법

### Streamlit Cloud 배포

1. GitHub에 프로젝트 푸시
2. https://streamlit.io/cloud 접속
3. "New app" 클릭
4. 저장소 선택 및 `app.py` 파일 지정
5. Secrets에 API 키 추가:
   - `OPENAI_API_KEY`: your_key
   - `TAVILY_API_KEY`: your_key
6. Deploy!

### 로컬 네트워크 공유

같은 네트워크의 다른 사람이 접근하려면:

```bash
streamlit run app.py --server.address 0.0.0.0
```

다른 사람은 `http://<your-ip>:8501`로 접속할 수 있습니다.

## 🛠️ 문제 해결

### API 키 오류
- `.env` 파일이 프로젝트 루트에 있는지 확인
- API 키가 올바르게 입력되었는지 확인
- `python check_api_keys.py`로 테스트

### 벡터 스토어 오류
- `chroma_db` 폴더가 생성되었는지 확인
- PDF 문서를 먼저 추가해야 검색 가능

### 의존성 오류
```bash
pip install --upgrade -r requirements.txt
```

## 📝 라이선스

이 프로젝트는 자유롭게 사용할 수 있습니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요!

---

**만든이**: LangGraph RAG 에이전트  
**버전**: 1.0.0
