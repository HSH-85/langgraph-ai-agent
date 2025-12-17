# 🤖 고도화된 LangGraph RAG 에이전트

LangGraph 기반의 고도화된 RAG (Retrieval-Augmented Generation) 에이전트입니다.
의도 분석, 문서 리랭크, 무한 루프 방지, 사고 과정 추적 등 고급 기능을 포함합니다.

## ✨ 주요 기능

### 핵심 기능
- **🧠 의도 분석**: 사용자 질문의 의도를 4가지 유형으로 분류
  - `factual`: 사실적 정보 검색
  - `analytical`: 분석/비교 요구
  - `conversational`: 일반 대화
  - `procedural`: 절차/방법 설명
  
- **🔍 벡터 스토어 검색**: ChromaDB를 사용한 효율적인 문서 검색
- **🎯 리랭크**: Cohere Rerank API로 관련성 순 문서 재정렬
- **📊 관련성 평가**: LLM 기반 문서 관련성 자동 평가
- **🌐 웹 검색**: Tavily를 통한 실시간 웹 검색 (관련성 낮을 시 자동 실행)
- **📝 맞춤형 답변**: 의도에 따른 최적화된 답변 생성

### 고도화 기능
- **🔁 무한 루프 방지**: 최대 3회 재시도로 안정성 보장
- **🧭 사고 과정 추적**: 각 단계별 에이전트 내부 로직 가시화
- **📈 메타 정보 표시**: 의도, 관련성, 문서 수 등 상세 정보 제공
- **💬 대화형 UI**: Streamlit 기반 직관적 사용자 인터페이스

## 🏗️ 시스템 아키텍처

```
사용자 질문
    ↓
[analyze_intent] - 질문 의도 분석
    ↓ (factual/analytical/conversational/procedural)
[retrieve] - 벡터 스토어에서 문서 검색 (k=5)
    ↓
[rerank] - Cohere로 문서 재정렬 (top 3 선택)
    ↓
[grade_documents] - LLM으로 관련성 평가
    ↓
[조건 분기] (loop_count < 3 검사)
    ├─ 관련성 높음 → [generate] - 의도별 맞춤 답변 생성
    └─ 관련성 낮음 → [web_search] - 웹 검색 → [generate]
         ↓
    (재시도 카운터 증가, 최대 3회)
```

### 상태 관리 스키마

```python
AgentState:
  - messages: 대화 기록 (List[BaseMessage])
  - question: 사용자 질문 (str)
  - intent: 의도 분석 결과 (Optional[str])
  - documents: 검색된 문서 리스트 (List[str])
  - context: 최종 컨텍스트 (Optional[str])
  - is_relevant: 관련성 평가 ('yes'/'no'/'partial')
  - web_search: 웹 검색 필요 여부 (bool)
  - loop_count: 재시도 횟수 (int, 최대 3)
  - current_step: 현재 실행 단계 (Optional[str])
  - thought_process: 사고 과정 로그 (List[str])
```

## 🚀 빠른 시작

### 1. 저장소 클론 또는 다운로드

```bash
git clone https://github.com/HSH-85/langgraph-ai-agent.git
cd langgraph-ai-agent
```

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
COHERE_API_KEY=your_cohere_api_key_here
```

**API 키 발급:**
- OpenAI API 키: https://platform.openai.com/api-keys
- Tavily API 키: https://tavily.com (웹 검색용, 선택사항)
- Cohere API 키: https://cohere.com (리랭크용, 선택사항)

| 변수명 | 설명 | 필수 여부 |
|--------|------|----------|
| `OPENAI_API_KEY` | OpenAI API 키 | ✅ 필수 |
| `TAVILY_API_KEY` | Tavily API 키 (웹 검색용) | ⚠️ 선택 |
| `COHERE_API_KEY` | Cohere API 키 (리랭크용) | ⚠️ 선택 |

### 5. API 키 확인 (선택사항)

```bash
python check_api_keys.py
```

### 6. 실행

**Streamlit UI (권장):**
```bash
streamlit run app.py
```

**CLI 모드 (테스트용):**
```bash
python main.py
```

## 💻 사용 방법

### 1. Streamlit UI (권장)

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하여 사용할 수 있습니다.

**UI 기능:**
- 💬 실시간 채팅 인터페이스
- 🧠 사고 과정 표시 토글 (각 단계별 로그 확인)
- 📊 메타 정보 표시 토글 (의도, 관련성, 문서 수 등)
- ✅ API 키 상태 확인

### 2. CLI 모드 (테스트용)

```bash
python main.py
```

**출력 정보:**
- 사고 과정 (각 단계별 로그)
- 최종 답변
- 메타 정보 (의도, 관련성, 재시도 횟수 등)

## 📁 프로젝트 구조

```
aiAgent/
├── main.py              # 고도화된 LangGraph 워크플로우 정의 및 실행
├── nodes.py             # 노드 함수 (analyze_intent, retrieve, rerank, grade, search, generate)
├── state.py             # AgentState 스키마 (고도화된 상태 관리)
├── app.py               # Streamlit UI (사고 과정 추적 포함)
├── check_api_keys.py    # API 키 확인 유틸리티
├── requirements.txt     # 의존성 패키지
├── .env                 # 환경 변수 (API 키)
├── .gitignore          # Git 제외 파일
├── README.md           # 프로젝트 문서
├── README_ADVANCED.md  # 고도화 기능 설명 (이 파일)
├── DEPLOY.md           # 배포 가이드
└── chroma_db/          # ChromaDB 데이터 (자동 생성)
```

### 주요 파일 설명

#### `state.py`
`AgentState` 스키마 정의 - 고도화된 상태 관리:
- 의도 분석 결과 저장
- 무한 루프 방지를 위한 카운터
- 사고 과정 추적용 로그 리스트

#### `nodes.py`
노드 함수 구현:
- **`analyze_intent`**: 질문 의도 분석 (LLM 기반)
- **`retrieve`**: 벡터 검색 (ChromaDB, k=5)
- **`rerank_documents`**: Cohere 리랭크 (top 3 선택)
- **`grade_documents`**: 관련성 평가 (LLM 기반, 무한 루프 방지)
- **`web_search_node`**: 웹 검색 (Tavily)
- **`generate`**: 의도별 맞춤 답변 생성

#### `main.py`
노드 연결 및 워크플로우 컴파일:
- 조건부 분기 (웹 검색 여부)
- 무한 루프 방지 로직
- 테스트 코드 포함

#### `app.py`
Streamlit UI:
- 사고 과정 표시 (각 노드별 로그)
- 메타 정보 표시 (의도, 관련성, 문서 수 등)
- API 키 상태 확인

## 🎯 사용 예시

### 예시 1: 사실적 정보 검색

**질문:** "LangGraph란 무엇인가요?"

**사고 과정:**
```
🧠 의도 분석: factual
📚 벡터 스토어 검색: 0개 문서 발견
🎯 리랭크: 문서 없음
❌ 문서 평가: 문서 없음 → 웹 검색 필요
🌐 웹 검색: 3개 결과 추가
📝 답변 생성: 3개 문서 기반
✅ 답변 생성 완료
```

**메타 정보:**
- 의도: factual
- 관련성: no (벡터 스토어에 문서 없음)
- 재시도 횟수: 1
- 문서 개수: 3 (웹 검색 결과)

### 예시 2: 분석/비교 요구

**질문:** "LangChain과 LangGraph의 차이점은?"

**의도:** analytical

**답변 스타일:** 비교, 분석, 평가를 통해 다각도로 답변하고, 장단점을 균형있게 제시

### 예시 3: 절차/방법 설명

**질문:** "LangGraph로 RAG 에이전트를 만드는 방법은?"

**의도:** procedural

**답변 스타일:** 단계별로 명확하게 설명하고, 실행 가능한 방법을 제시

## 🔧 고도화 세부 사항

### 1. 의도 분석 시스템

사용자 질문을 4가지 유형으로 분류:

```python
intents = {
    'factual': "사실적 정보 검색",
    'analytical': "비교/분석 요구",
    'conversational': "일반 대화",
    'procedural': "절차/방법 설명"
}
```

각 의도에 맞는 답변 스타일 적용.

### 2. 무한 루프 방지

```python
if loop_count >= 3:
    # 강제로 답변 생성 단계로 이동
    return "generate"
```

최대 3회 재시도로 무한 루프 방지.

### 3. 사고 과정 추적

각 노드에서 `thought_process` 리스트에 로그 추가:

```python
thought_process.append("🧠 의도 분석: factual")
thought_process.append("📚 벡터 스토어 검색: 5개 문서 발견")
thought_process.append("🎯 리랭크: 5개 → 3개 (상위 문서 선택)")
```

UI에서 사고 과정 표시 가능.

### 4. 메타 정보 관리

상태에 메타 정보 저장:

```python
metadata = {
    "intent": "factual",
    "is_relevant": "yes",
    "doc_count": 3,
    "loop_count": 0,
    "web_search_used": False
}
```

UI에서 메타 정보 표시 가능.

## 🛠️ 문제 해결

### 1. API 키 오류

```bash
python check_api_keys.py
```

API 키가 올바르게 로드되는지 확인.

### 2. Cohere API 키 없음

리랭크 기능이 비활성화되지만, 나머지 기능은 정상 작동합니다.

### 3. Tavily API 키 없음

웹 검색 기능이 비활성화되지만, 벡터 스토어 검색은 정상 작동합니다.

### 4. ChromaDB 오류

```bash
# ChromaDB 데이터 삭제 후 재생성
rm -rf chroma_db/
```

### 5. 인코딩 오류 (Windows)

`main.py`에서 UTF-8 인코딩 설정이 자동으로 적용됩니다.

## 📊 성능 최적화

### 1. 검색 개수 조정

```python
# nodes.py의 retrieve 함수
docs = vectorstore.similarity_search(question, k=5)  # k 값 조정
```

### 2. 리랭크 top_n 조정

```python
# nodes.py의 rerank_documents 함수
rerank_response = cohere_client.rerank(
    model="rerank-multilingual-v3.0",
    query=question,
    documents=documents,
    top_n=min(len(documents), 3)  # top_n 값 조정
)
```

### 3. 재시도 횟수 조정

```python
# nodes.py의 grade_documents 함수
if loop_count >= 3:  # 최대 재시도 횟수 조정
    # ...
```

## 🤝 기여

기여는 언제나 환영입니다! 이슈나 PR을 자유롭게 올려주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🔗 관련 링크

- [LangChain 문서](https://python.langchain.com/)
- [LangGraph 문서](https://langchain-ai.github.io/langgraph/)
- [Cohere Rerank 문서](https://docs.cohere.com/docs/reranking)
- [Tavily API 문서](https://docs.tavily.com/)
- [Streamlit 문서](https://docs.streamlit.io/)

## 📮 문의

문제가 발생하거나 질문이 있으시면 GitHub Issues를 이용해 주세요.

---

**고도화된 LangGraph RAG 에이전트 | Powered by OpenAI, Tavily & Cohere**


