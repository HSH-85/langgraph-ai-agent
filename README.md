# 🤖 고도화된 LangGraph RAG 에이전트 (금융 특화)

LangGraph를 기반으로 한 고도화된 RAG(Retrieval-Augmented Generation) 에이전트입니다. 금융 도메인에 특화된 분석 기능과 다중 검색/검증 시스템을 통해 정확하고 신뢰할 수 있는 답변을 생성합니다.

## ✨ 주요 기능

### 🧠 핵심 RAG 기능
- **🔍 벡터 스토어 검색**: ChromaDB를 사용한 문서 검색
- **🎯 리랭크**: Cohere를 사용한 문서 재정렬 (관련성 순 정렬)
- **🌐 웹 검색**: Tavily를 통한 실시간 웹 검색
- **📝 답변 생성**: OpenAI GPT 모델을 사용한 답변 생성
- **💬 대화형 UI**: Streamlit 기반 사용자 인터페이스
- **🧠 지능형 라우팅**: 문서 관련성 평가 후 웹 검색 여부 자동 결정

### 💰 금융 특화 기능
- **도메인 분석**: 주식, 채권, 외환, 부동산, 금리, 파생상품, 암호화폐, 경제지표 자동 분류
- **의도 분석**: 사실적 정보, 분석/비교, 일반 대화, 절차 설명 등 의도 자동 파악
- **다중 검색 라운드**: 최대 3라운드까지 반복 검색으로 정보 수집
- **다중 검증 라운드**: 최대 2라운드까지 검증으로 정확도 향상
- **크로스 검증**: 여러 소스 간 정보 일치도 확인
- **신뢰도 계산**: 답변의 신뢰도 점수 자동 계산 (0.0 ~ 1.0)
- **회사 비교 분석**: 비슷한 업종/규모의 회사 자동 비교
- **사고 과정 추적**: 에이전트의 사고 과정을 단계별로 표시
- **메타 정보 제공**: 의도, 관련성, 신뢰도, 소스 일치도 등 상세 정보

## 📁 프로젝트 구조

```
aiAgent/
├── main.py          # LangGraph 그래프 정의 및 실행 함수
├── nodes.py         # 노드 함수들 (의도 분석, 도메인 분석, 검색, 검증, 답변 생성 등)
├── state.py         # 상태 정의 (AgentState - 고도화된 상태 관리)
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
COHERE_API_KEY=your_cohere_api_key_here
```

**API 키 발급:**
- **OpenAI API 키**: https://platform.openai.com/api-keys (필수)
- **Tavily API 키**: https://tavily.com (웹 검색용, 선택사항)
- **Cohere API 키**: https://cohere.com (리랭크용, 선택사항)

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
2. 브라우저에서 질문 입력 (예: "삼성전자 주가는 얼마인가요?")
3. 에이전트가 자동으로:
   - 사용자 의도 분석
   - 금융 도메인 분류
   - 벡터 스토어에서 문서 검색
   - 문서 관련성 평가
   - 필요시 웹 검색 수행 (최대 3라운드)
   - 크로스 검증 및 신뢰도 계산
   - 회사 비교 분석 (주식 질문의 경우)
   - 최종 답변 생성

### UI 기능

- **사고 과정 보기**: 에이전트의 단계별 사고 과정 확인
- **메타 정보 보기**: 의도, 관련성, 신뢰도, 소스 일치도 등 상세 정보 확인
- **대화 히스토리**: 이전 대화 내용 자동 저장 및 활용

### 문서 추가하기

PDF 문서를 벡터 스토어에 추가하려면:

```python
from nodes import load_pdf_to_vectorstore

# PDF 파일을 벡터 스토어에 추가
retriever = load_pdf_to_vectorstore("path/to/document.pdf")
```

## 🎯 주요 워크플로우

```
사용자 질문
    ↓
의도 분석 (factual/analytical/conversational/procedural)
    ↓
금융 도메인 분석 (stock/bond/forex/economic 등)
    ↓
벡터 스토어 검색
    ↓
리랭크 (Cohere)
    ↓
문서 관련성 평가
    ↓
[관련성 낮음] → 웹 검색 (최대 3라운드)
    ↓
크로스 검증 (소스 일치도 확인)
    ↓
신뢰도 계산
    ↓
[주식 질문] → 회사 비교 분석
    ↓
금융 특화 답변 생성
    ↓
최종 답변 + 메타 정보
```

## 🔧 환경 변수

| 변수명 | 설명 | 필수 여부 |
|--------|------|----------|
| `OPENAI_API_KEY` | OpenAI API 키 (GPT 모델 사용) | ✅ 필수 |
| `TAVILY_API_KEY` | Tavily API 키 (웹 검색용) | ⚠️ 선택 (웹 검색 기능 사용 시) |
| `COHERE_API_KEY` | Cohere API 키 (리랭크용) | ⚠️ 선택 (리랭크 기능 사용 시) |

## 💡 사용 예시

### 주식 관련 질문
```
질문: "삼성전자 주가는 얼마인가요?"
→ 도메인: 주식
→ 웹 검색으로 최신 주가 정보 수집
→ 비슷한 업종 회사 비교 분석
→ 신뢰도 점수와 함께 답변 제공
```

### 경제 지표 질문
```
질문: "현재 기준금리는 얼마인가요?"
→ 도메인: 금리
→ 웹 검색으로 최신 금리 정보 수집
→ 크로스 검증으로 정보 정확도 확인
```

### 분석 요청
```
질문: "삼성전자와 SK하이닉스를 비교해주세요"
→ 의도: analytical
→ 도메인: 주식
→ 두 회사 정보 수집 및 비교 분석
```

## 🌐 배포 방법

### Streamlit Cloud 배포

1. GitHub에 프로젝트 푸시
2. https://streamlit.io/cloud 접속
3. "New app" 클릭
4. 저장소 선택 및 `app.py` 파일 지정
5. Secrets에 API 키 추가:
   - `OPENAI_API_KEY`: your_key
   - `TAVILY_API_KEY`: your_key (선택)
   - `COHERE_API_KEY`: your_key (선택)
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
- 문서가 없어도 웹 검색으로 답변 가능

### 의존성 오류
```bash
pip install --upgrade -r requirements.txt
```

### 웹 검색이 작동하지 않는 경우
- `TAVILY_API_KEY`가 설정되었는지 확인
- API 키가 유효한지 확인
- 벡터 스토어에 관련 문서가 있으면 웹 검색 없이도 답변 가능

## 📊 성능 및 특징

- **다중 검색 라운드**: 정보가 부족하면 자동으로 추가 검색 (최대 3라운드)
- **크로스 검증**: 여러 소스의 정보를 비교하여 정확도 향상
- **신뢰도 점수**: 답변의 신뢰도를 0.0 ~ 1.0 점수로 제공
- **무한 루프 방지**: 최대 7회까지 재시도로 안정성 보장
- **대화 컨텍스트**: 이전 대화 내용을 활용한 연속 질문 지원

## 🔄 업데이트 내역

### v1.0.0 (현재 버전)
- ✅ 기본 RAG 기능 구현
- ✅ 금융 도메인 특화 기능 추가
- ✅ 다중 검색/검증 라운드 구현
- ✅ 신뢰도 계산 및 크로스 검증
- ✅ 회사 비교 분석 기능
- ✅ 사고 과정 추적 및 메타 정보 제공
- ✅ Streamlit UI 구현

## 📝 라이선스

이 프로젝트는 자유롭게 사용할 수 있습니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요!

---

**만든이**: LangGraph RAG 에이전트  
**버전**: 1.0.0  
**특화 도메인**: 금융 (주식, 채권, 외환, 경제지표 등)
