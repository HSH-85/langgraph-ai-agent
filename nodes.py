"""
LangGraph RAG 에이전트의 노드 함수들
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage
from tavily import TavilyClient
from state import AgentState
import cohere
import os
from datetime import datetime
from dotenv import load_dotenv

# 하위 호환성 유지
GraphState = AgentState

load_dotenv()

# API 키 확인
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# API 키 확인 (조용한 모드 - 필요시 주석 해제)
# if not openai_api_key:
#     print("⚠️ 경고: OPENAI_API_KEY가 설정되지 않았습니다.")
#     print("   .env 파일에 OPENAI_API_KEY를 설정해주세요.")

# LLM 초기화 (환경 변수에서 자동으로 API 키를 읽어옴)
# api_key 파라미터를 명시하지 않으면 OPENAI_API_KEY 환경 변수를 자동으로 사용
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0
)
embeddings = OpenAIEmbeddings()

# Tavily 클라이언트 초기화
if tavily_api_key:
    tavily_client = TavilyClient(api_key=tavily_api_key)
else:
    tavily_client = None
    # print("⚠️ 경고: TAVILY_API_KEY가 설정되지 않았습니다. 웹 검색 기능이 작동하지 않습니다.")

# Cohere 클라이언트 초기화 (리랭크용)
cohere_api_key = os.getenv("COHERE_API_KEY")
if cohere_api_key:
    cohere_client = cohere.Client(api_key=cohere_api_key)
else:
    cohere_client = None
    # print("⚠️ 경고: COHERE_API_KEY가 설정되지 않았습니다. 리랭크 기능이 작동하지 않습니다.")


# 전역 변수: Retriever (초기화 필요)
retriever = None


def set_retriever(new_retriever):
    """Retriever를 설정하는 함수"""
    global retriever
    retriever = new_retriever


def analyze_intent(state: AgentState) -> AgentState:
    """
    사용자 질문의 의도를 분석하는 노드 (고도화)
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 (intent, current_step, thought_process)
    """
    question = state.get("question", "")
    thought_process = state.get("thought_process", [])
    
    # 의도 분석 프롬프트
    intent_prompt = f"""다음 질문의 의도를 분석해주세요.

질문: {question}

의도 분류:
1. 'factual': 사실적 정보나 지식을 묻는 질문 (예: "파이썬이란?", "LangGraph 특징은?")
2. 'analytical': 비교, 분석, 평가를 요구하는 질문 (예: "A와 B의 차이는?", "장단점은?")
3. 'conversational': 일반 대화나 의견 교환 (예: "안녕", "고마워")
4. 'procedural': 절차, 방법, 사용법을 묻는 질문 (예: "어떻게 하나요?", "설치 방법은?")

위 4가지 중 하나만 답변해주세요: """

    try:
        response = llm.invoke(intent_prompt)
        intent = response.content.strip().lower() if hasattr(response, 'content') else "factual"
        
        # 유효한 의도 값으로 정규화
        valid_intents = ['factual', 'analytical', 'conversational', 'procedural']
        if intent not in valid_intents:
            intent = 'factual'  # 기본값
            
        thought_process.append(f"🧠 의도 분석: {intent}")
    except Exception:
        intent = 'factual'
        thought_process.append("🧠 의도 분석: factual (기본값)")
    
    return {
        "intent": intent,
        "current_step": "문서 검색 중...",
        "thought_process": thought_process
    }


def retrieve(state: AgentState) -> AgentState:
    """
    질문을 받아 VectorDB에서 문서를 검색하는 노드
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 (documents 포함)
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    thought_process = state.get("thought_process", [])
    
    # Retriever가 설정되지 않은 경우 ChromaDB에서 직접 검색
    if retriever is None:
        try:
            vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
            docs = vectorstore.similarity_search(question, k=5)  # 리랭크를 위해 더 많이 검색
            documents = [doc.page_content for doc in docs]
            thought_process.append(f"📚 벡터 스토어 검색: {len(documents)}개 문서 발견")
        except Exception:
            # 벡터 스토어가 없거나 오류 발생 시 빈 리스트 반환 (조용히 처리)
            documents = []
            thought_process.append("📚 벡터 스토어 검색: 문서 없음")
    else:
        # Retriever를 사용하여 문서 검색
        try:
            docs = retriever.invoke(question)
            documents = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs]
            thought_process.append(f"📚 벡터 스토어 검색: {len(documents)}개 문서 발견")
        except Exception:
            # Retriever 검색 오류 시 빈 리스트 반환 (조용히 처리)
            documents = []
            thought_process.append("📚 벡터 스토어 검색: 문서 없음")
    
    return {
        "documents": documents,
        "current_step": "문서 리랭크 중...",
        "thought_process": thought_process
    }


def rerank_documents(state: AgentState) -> AgentState:
    """
    검색된 문서들을 리랭크하여 관련성 순으로 재정렬하는 노드 (고도화)
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 (재정렬된 documents)
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    thought_process = state.get("thought_process", [])
    
    # 문서가 없거나 Cohere 클라이언트가 없으면 그대로 반환
    if not documents or len(documents) == 0:
        thought_process.append("🎯 리랭크: 문서 없음")
        return {
            "documents": documents,
            "current_step": "문서 평가 중...",
            "thought_process": thought_process
        }
    
    if cohere_client is None:
        thought_process.append(f"🎯 리랭크: 스킵 (Cohere 미설정, {len(documents)}개 유지)")
        return {
            "documents": documents,
            "current_step": "문서 평가 중...",
            "thought_process": thought_process
        }
    
    try:
        # Cohere Rerank API를 사용하여 문서 재정렬
        # top_n: 상위 N개 문서만 반환 (기본값: 문서 전체)
        rerank_response = cohere_client.rerank(
            model="rerank-multilingual-v3.0",  # 다국어 지원 모델
            query=question,
            documents=documents,
            top_n=min(len(documents), 3)  # 최대 3개 문서만 반환
        )
        
        # 재정렬된 문서 추출
        # documents가 문자열 리스트이므로 result.document도 문자열입니다
        reranked_documents = [
            result.document['text'] if isinstance(result.document, dict) and 'text' in result.document
            else str(result.document) if not isinstance(result.document, str)
            else result.document
            for result in rerank_response.results
        ]
        
        thought_process.append(f"🎯 리랭크: {len(documents)}개 → {len(reranked_documents)}개 (상위 문서 선택)")
        
        return {
            "documents": reranked_documents,
            "current_step": "문서 평가 중...",
            "thought_process": thought_process
        }
    except Exception as e:
        # 리랭크 실패 시 원본 문서 반환
        thought_process.append(f"🎯 리랭크: 실패 ({len(documents)}개 유지)")
        return {
            "documents": documents,
            "current_step": "문서 평가 중...",
            "thought_process": thought_process
        }


def grade_documents(state: AgentState) -> AgentState:
    """
    검색된 문서가 질문과 관련 있는지 LLM(gpt-4o-mini)으로 평가하는 노드 (고도화)
    관련 없으면 web_search=True 설정
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 (web_search, is_relevant, loop_count 포함)
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    thought_process = state.get("thought_process", [])
    loop_count = state.get("loop_count", 0)
    
    # 무한 루프 방지: 금융 특화 최대 7회까지만 재시도
    if loop_count >= 7:
        thought_process.append(f"⚠️ 최대 재시도 횟수 도달 ({loop_count}회)")
        return {
            "web_search": False,
            "is_relevant": "no",
            "loop_count": loop_count,
            "current_step": "답변 생성 중...",
            "thought_process": thought_process
        }
    
    # 문서가 없으면 웹 검색 필요
    if not documents or len(documents) == 0:
        thought_process.append("❌ 문서 평가: 문서 없음 → 웹 검색 필요")
        return {
            "web_search": True,
            "is_relevant": "no",
            "loop_count": loop_count + 1,
            "current_step": "웹 검색 중...",
            "thought_process": thought_process
        }
    
    # 문서들을 하나의 문자열로 결합
    documents_text = "\n\n".join([f"[문서 {i+1}]\n{doc}" for i, doc in enumerate(documents)])
    
    # LLM을 사용하여 문서 관련성 평가
    evaluation_prompt = f"""다음 질문과 검색된 문서들을 검토하고, 문서들이 질문에 답변하기에 충분한 관련성이 있는지 평가해주세요.

질문: {question}

검색된 문서들:
{documents_text}

평가 기준:
- 문서들이 질문과 직접적으로 관련이 있고 답변에 충분한 정보를 제공하는 경우: "yes"
- 문서들이 질문과 관련이 없거나 답변에 필요한 정보가 부족한 경우: "no"

답변은 반드시 "yes" 또는 "no"로만 답변해주세요."""

    try:
        response = llm.invoke(evaluation_prompt)
        evaluation = response.content.strip().lower() if hasattr(response, 'content') else str(response).strip().lower()
        
        # "yes"가 아니면 웹 검색 필요
        web_search_needed = not (evaluation.startswith("yes") or evaluation == "yes")
        is_relevant = "yes" if not web_search_needed else "no"
        
        if web_search_needed:
            thought_process.append(f"❌ 문서 평가: 관련성 낮음 → 웹 검색 필요 ({loop_count + 1}회 시도)")
        else:
            thought_process.append("✅ 문서 평가: 관련성 높음 → 답변 생성")
            
    except Exception as e:
        # 오류 발생 시 웹 검색으로 폴백 (조용히 처리)
        web_search_needed = True
        is_relevant = "no"
        thought_process.append(f"⚠️ 문서 평가: 오류 발생 → 웹 검색 실행 ({loop_count + 1}회 시도)")
    
    return {
        "web_search": web_search_needed,
        "is_relevant": is_relevant,
        "loop_count": loop_count + 1 if web_search_needed else loop_count,
        "current_step": "웹 검색 중..." if web_search_needed else "답변 생성 중...",
        "thought_process": thought_process
    }


def web_search_node(state: AgentState) -> AgentState:
    """
    web_search=True일 때 Tavily API로 웹 검색을 수행하고 결과를 문서에 추가하는 노드 (고도화)
    현재 시간 정보를 검색 쿼리에 포함하여 최신 정보를 검색합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 (documents에 웹 검색 결과 추가)
    """
    question = state.get("question", "")
    existing_documents = state.get("documents", [])
    thought_process = state.get("thought_process", [])
    search_round = state.get("search_round", 0)
    financial_domain = state.get("financial_domain", "general")
    
    if tavily_client is None:
        thought_process.append("⚠️ 웹 검색: Tavily 미설정 → 스킵")
        return {
            "documents": existing_documents,
            "web_search": False,
            "current_step": "답변 생성 중...",
            "thought_process": thought_process
        }
    
    try:
        # 현재 시간 정보 가져오기
        now = datetime.now()
        current_date = now.strftime("%Y년 %m월 %d일")
        current_time = now.strftime("%H시 %M분")
        current_datetime_str = f"{current_date} {current_time}"
        
        # 다중 검색 라운드: 각 라운드마다 다른 쿼리 생성
        domain_keywords = {
            "stock": ["주가", "시가총액", "PER", "PBR", "기업 실적"],
            "bond": ["수익률", "만기", "신용등급", "이자율"],
            "forex": ["환율", "환차익", "통화정책"],
            "real_estate": ["부동산 가격", "전세", "월세", "부동산 시장"],
            "interest_rate": ["금리", "기준금리", "금리 정책"],
            "derivative": ["파생상품", "옵션", "선물"],
            "crypto": ["암호화폐", "비트코인", "가상자산"],
            "economic": ["경제 지표", "GDP", "인플레이션"],
        }
        
        # 검색 라운드별 쿼리 생성
        if search_round == 0:
            enhanced_query = f"{question} (현재 시간: {current_datetime_str})"
        elif search_round == 1:
            keywords = domain_keywords.get(financial_domain, [])
            additional = f" {keywords[0]}" if keywords else ""
            enhanced_query = f"{question}{additional} 최신 동향 (현재 시간: {current_datetime_str})"
        elif search_round == 2:
            # 3라운드: 업종/경쟁사 비교 정보 검색 (주식 도메인인 경우)
            if financial_domain == "stock":
                enhanced_query = f"{question} 업종 경쟁사 비교 분석 (현재 시간: {current_datetime_str})"
            else:
                enhanced_query = f"{question} 상세 분석 최신 정보 (현재 시간: {current_datetime_str})"
        else:
            enhanced_query = f"{question} 상세 분석 최신 정보 (현재 시간: {current_datetime_str})"
        
        # Tavily 검색
        response = tavily_client.search(
            query=enhanced_query,
            max_results=3,
            search_depth="advanced"
        )
        
        # 검색 결과를 문서에 추가 (현재 시간 정보 포함)
        search_results = []
        for result in response.get("results", []):
            content = result.get("content", "")
            if content:
                # 각 검색 결과에 현재 시간 정보 추가
                timestamped_content = f"[검색 시간: {current_datetime_str}, 검색 라운드: {search_round + 1}]\n{content}"
                search_results.append(timestamped_content)
        
        existing_documents.extend(search_results)
        thought_process.append(f"🌐 웹 검색 ({search_round + 1}라운드): {len(search_results)}개 결과 추가 (검색 시간: {current_datetime_str})")
    except Exception as e:
        # 웹 검색 오류 시 기존 문서 유지 (조용히 처리)
        thought_process.append("⚠️ 웹 검색: 오류 발생")
    
    return {
        "documents": existing_documents,
        "search_round": search_round + 1,  # 검색 라운드 증가
        "web_search": False,  # 웹 검색 완료 후 플래그 리셋
        "current_step": "크로스 검증 중...",
        "thought_process": thought_process
    }


def generate(state: AgentState) -> AgentState:
    """
    확보된 문서들을 context로 삼아 최종 답변을 생성하는 노드 (고도화)
    현재 시간 정보를 컨텍스트에 포함하여 정확한 시간 정보를 제공합니다.
    이전 대화 맥락을 고려하여 답변합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 (messages, context에 답변 추가)
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    messages = state.get("messages", [])
    intent = state.get("intent", "factual")
    thought_process = state.get("thought_process", [])
    
    # 현재 시간 정보 가져오기
    now = datetime.now()
    current_date = now.strftime("%Y년 %m월 %d일")
    current_time = now.strftime("%H시 %M분")
    current_datetime_str = f"{current_date} {current_time}"
    current_weekday = now.strftime("%A")  # 영어 요일
    weekday_kr = {
        'Monday': '월요일',
        'Tuesday': '화요일',
        'Wednesday': '수요일',
        'Thursday': '목요일',
        'Friday': '금요일',
        'Saturday': '토요일',
        'Sunday': '일요일'
    }
    current_weekday_kr = weekday_kr.get(current_weekday, current_weekday)
    full_datetime_str = f"{current_date} {current_weekday_kr} {current_time}"
    
    # 문서들을 컨텍스트로 결합
    if documents:
        context_text = "\n\n".join([f"[문서 {i+1}]\n{doc}" for i, doc in enumerate(documents)])
        thought_process.append(f"📝 답변 생성: {len(documents)}개 문서 기반")
    else:
        context_text = "관련 정보를 찾을 수 없습니다."
        thought_process.append("📝 답변 생성: 문서 없음 (일반 응답)")
    
    # 의도에 따른 프롬프트 조정
    intent_instructions = {
        'factual': "정확한 사실을 제공하고, 출처가 명확한 정보를 우선하여 답변하세요.",
        'analytical': "비교, 분석, 평가를 통해 다각도로 답변하고, 장단점을 균형있게 제시하세요.",
        'conversational': "자연스럽고 친근한 어조로 답변하세요.",
        'procedural': "단계별로 명확하게 설명하고, 실행 가능한 방법을 제시하세요."
    }
    intent_instruction = intent_instructions.get(intent, intent_instructions['factual'])
    
    # 이전 대화 맥락 구성 (최근 3개 대화만 포함하여 토큰 절약)
    conversation_context = ""
    if messages and len(messages) > 0:
        # 최근 대화만 추출 (HumanMessage, AIMessage 쌍)
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        conversation_parts = []
        for msg in recent_messages:
            if hasattr(msg, 'content'):
                role = "사용자" if hasattr(msg, '__class__') and "Human" in msg.__class__.__name__ else "어시스턴트"
                conversation_parts.append(f"[{role}]: {msg.content}")
        if conversation_parts:
            conversation_context = "\n\n[이전 대화 맥락]\n" + "\n".join(conversation_parts) + "\n\n"
    
    # 프롬프트 구성 (현재 시간 정보 및 이전 대화 맥락 포함)
    prompt = f"""다음 컨텍스트를 기반으로 질문에 답변해주세요.
이전 대화 맥락을 고려하여 자연스럽고 연관성 있는 답변을 제공하세요.

[중요: 현재 시간 정보]
현재 시간은 {full_datetime_str}입니다.
시간 관련 질문이 있다면 이 정보를 기준으로 답변하세요.

{conversation_context}
[답변 스타일]
{intent_instruction}

[검색된 문서 컨텍스트]
{context_text}

[현재 질문]
{question}

[답변] (이전 대화와 자연스럽게 연결되도록 답변하세요)
"""

    # LLM을 사용하여 답변 생성
    try:
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        thought_process.append("✅ 답변 생성 완료")
    except Exception as e:
        error_msg = str(e)
        
        # 더 자세한 에러 메시지 제공
        if "insufficient_quota" in error_msg or "429" in error_msg:
            answer = """⚠️ OpenAI API 할당량 오류가 발생했습니다.

가능한 원인:
1. OpenAI 계정에 크레딧이 없습니다
   → https://platform.openai.com/account/billing 에서 크레딧을 확인하세요
2. 무료 티어의 경우 초기 크레딧이 없을 수 있습니다
   → 결제 수단을 추가하거나 크레딧을 충전하세요
3. API 키가 잘못되었거나 다른 계정의 키일 수 있습니다
   → .env 파일의 OPENAI_API_KEY를 확인하세요"""
        elif "api_key" in error_msg.lower():
            answer = """⚠️ OpenAI API 키 오류가 발생했습니다.

.env 파일에 올바른 OPENAI_API_KEY를 설정해주세요.
API 키는 https://platform.openai.com/api-keys 에서 발급받을 수 있습니다."""
        else:
            answer = f"죄송합니다. 답변을 생성하는 중 오류가 발생했습니다.\n\n오류 내용: {error_msg}"
    
    # 메시지에 추가
    updated_messages = messages + [
        HumanMessage(content=question),
        AIMessage(content=answer)
    ]
    
    return {
        "messages": updated_messages,
        "context": context_text,
        "current_step": "완료",
        "thought_process": thought_process
    }


def analyze_financial_domain(state: AgentState) -> AgentState:
    """
    금융 도메인 분석 노드
    질문을 금융 도메인으로 분류합니다 (주식, 채권, 외환, 부동산 등)
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 (financial_domain 포함)
    """
    question = state.get("question", "")
    thought_process = state.get("thought_process", [])
    
    financial_domain_prompt = f"""다음 질문을 금융 도메인으로 분류해주세요.

질문: {question}

금융 도메인 분류 (하나만 선택):
- 'stock': 주식, 주가, 기업 분석, 시가총액, PER, PBR, 배당 등 (회사명 + 주가/주식 관련 질문은 반드시 stock 선택)
- 'bond': 채권, 이자율, 수익률, 만기, 신용등급 등
- 'forex': 외환, 환율, 통화정책, 환차익 등
- 'real_estate': 부동산, 집값, 전세, 월세, 부동산 투자 등
- 'interest_rate': 금리, 기준금리, 금리 정책 등
- 'derivative': 파생상품, 옵션, 선물, 스왑 등
- 'crypto': 암호화폐, 가상자산, 비트코인, 이더리움 등
- 'economic': 경제 지표, GDP, 인플레이션, 실업률, 경기 등
- 'general': 일반 금융, 금융 상품, 금융 서비스 등

중요: 회사명이 포함된 주가/주식 관련 질문은 반드시 'stock'으로 분류하세요.
예: "삼성전자 주가는?" → stock
예: "애플 주가" → stock
예: "SK하이닉스 주식" → stock

위 목록 중 하나만 답변해주세요: """
    
    try:
        response = llm.invoke(financial_domain_prompt)
        domain = response.content.strip().lower() if hasattr(response, 'content') else "general"
        
        # 유효한 도메인 값으로 정규화
        valid_domains = ['stock', 'bond', 'forex', 'real_estate', 'interest_rate', 
                        'derivative', 'crypto', 'economic', 'general']
        if domain not in valid_domains:
            domain = 'general'
        
        domain_kr = {
            'stock': '주식',
            'bond': '채권',
            'forex': '외환',
            'real_estate': '부동산',
            'interest_rate': '금리',
            'derivative': '파생상품',
            'crypto': '암호화폐',
            'economic': '경제 지표',
            'general': '일반 금융'
        }
        
        thought_process.append(f"💰 금융 도메인 분석: {domain_kr.get(domain, domain)}")
    except Exception:
        domain = 'general'
        thought_process.append("💰 금융 도메인 분석: 일반 금융 (기본값)")
    
    return {
        "financial_domain": domain,
        "current_step": "문서 검색 중...",
        "thought_process": thought_process
    }


def verify_documents(state: AgentState) -> AgentState:
    """
    금융 문서 검증 노드 (2차 검증)
    데이터 일관성, 출처 신뢰성, 시점 적절성, 상충 정보 확인
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 (verification_round 증가)
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    thought_process = state.get("thought_process", [])
    verification_round = state.get("verification_round", 0)
    
    if not documents or len(documents) == 0:
        thought_process.append("🔍 문서 검증: 문서 없음")
        return {
            "verification_round": verification_round + 1,
            "thought_process": thought_process
        }
    
    documents_text = "\n\n".join([f"[문서 {i+1}]\n{doc[:1000]}..." if len(doc) > 1000 else f"[문서 {i+1}]\n{doc}" 
                                  for i, doc in enumerate(documents)])
    
    verification_prompt = f"""다음 금융 관련 질문과 문서들을 검증해주세요:

질문: {question}

문서들:
{documents_text}

검증 항목:
1. 데이터 일관성: 숫자, 날짜, 통계, 비율이 문서 간 일치하는가?
2. 출처 신뢰성: 신뢰할 수 있는 금융 기관/미디어/데이터 소스인가?
3. 시점 적절성: 최신 정보인가? (금융 시장은 빠르게 변화함)
4. 상충 정보: 문서 간 모순이나 충돌하는 정보가 있는가?

각 문서에 대해 평가하고, 전체적으로:
- "verified": 검증 완료, 신뢰할 수 있음
- "needs_cross_check": 추가 확인 필요
- "unreliable": 신뢰할 수 없음

검증 결과를 한 단어로만 답변해주세요: """
    
    try:
        response = llm.invoke(verification_prompt)
        result = response.content.strip().lower() if hasattr(response, 'content') else "needs_cross_check"
        
        if "verified" in result or "검증 완료" in result:
            thought_process.append(f"✅ 문서 검증 ({verification_round + 1}라운드): 검증 완료")
        elif "unreliable" in result or "신뢰할 수 없" in result:
            thought_process.append(f"⚠️ 문서 검증 ({verification_round + 1}라운드): 신뢰도 낮음")
        else:
            thought_process.append(f"🔄 문서 검증 ({verification_round + 1}라운드): 추가 확인 필요")
    except Exception:
        thought_process.append(f"⚠️ 문서 검증 ({verification_round + 1}라운드): 오류 발생")
    
    return {
        "verification_round": verification_round + 1,
        "current_step": "문서 재평가 중...",
        "thought_process": thought_process
    }


def cross_validate(state: AgentState) -> AgentState:
    """
    크로스 검증 노드: 여러 소스 간 일치도 확인
    금융 정보의 정확성을 높이기 위해 여러 소스를 비교합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 (source_agreement, cross_validation_results 포함)
    """
    documents = state.get("documents", [])
    thought_process = state.get("thought_process", [])
    
    if len(documents) < 2:
        thought_process.append("🔄 크로스 검증: 소스 부족 (2개 미만)")
        return {
            "source_agreement": "low",
            "cross_validation_results": [],
            "thought_process": thought_process
        }
    
    # 문서 내용 요약 (너무 길면 잘라내기)
    docs_summary = "\n\n".join([f"[소스 {i+1}]\n{doc[:800]}..." if len(doc) > 800 else f"[소스 {i+1}]\n{doc}" 
                                for i, doc in enumerate(documents)])
    
    cross_validation_prompt = f"""다음 문서들은 같은 금융 질문에 대한 여러 소스입니다.
소스 간 일치도를 평가해주세요:

{docs_summary}

평가 기준:
- 핵심 데이터 (숫자, 비율, 통계 등)가 소스 간 일치하는가?
- 결론이나 해석이 일치하는가?
- 일치하는 정보의 비율은 얼마인가?

일치도를 다음 중 하나로 평가:
- "high": 80% 이상 일치
- "medium": 50-80% 일치
- "low": 50% 미만 일치

일치도만 한 단어로 답변해주세요: """
    
    try:
        response = llm.invoke(cross_validation_prompt)
        agreement = response.content.strip().lower() if hasattr(response, 'content') else "low"
        
        # 정규화
        if "high" in agreement or "높" in agreement or "80" in agreement:
            agreement = "high"
        elif "medium" in agreement or "중간" in agreement or "50" in agreement:
            agreement = "medium"
        else:
            agreement = "low"
        
        thought_process.append(f"🔄 크로스 검증: 소스 일치도 {agreement} ({len(documents)}개 소스)")
    except Exception:
        agreement = "low"
        thought_process.append("⚠️ 크로스 검증: 오류 발생")
    
    return {
        "source_agreement": agreement,
        "cross_validation_results": [{"agreement": agreement, "source_count": len(documents)}],
        "thought_process": thought_process
    }


def generate_financial(state: AgentState) -> AgentState:
    """
    금융 특화 답변 생성 노드
    금융 도메인에 특화된 정확하고 신뢰할 수 있는 답변을 생성합니다.
    비슷한 업종/규모의 회사 비교 분석 정보를 포함합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 (messages, context에 답변 추가)
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    messages = state.get("messages", [])
    intent = state.get("intent", "factual")
    financial_domain = state.get("financial_domain", "general")
    confidence_score = state.get("confidence_score", 0.0)
    source_agreement = state.get("source_agreement", "low")
    company_comparison_data = state.get("company_comparison_data")
    thought_process = state.get("thought_process", [])
    
    # 현재 시간 정보
    now = datetime.now()
    current_date = now.strftime("%Y년 %m월 %d일")
    current_time = now.strftime("%H시 %M분")
    current_weekday = now.strftime("%A")
    weekday_kr = {
        'Monday': '월요일', 'Tuesday': '화요일', 'Wednesday': '수요일',
        'Thursday': '목요일', 'Friday': '금요일', 'Saturday': '토요일', 'Sunday': '일요일'
    }
    current_weekday_kr = weekday_kr.get(current_weekday, current_weekday)
    full_datetime_str = f"{current_date} {current_weekday_kr} {current_time}"
    
    # 문서들을 컨텍스트로 결합
    if documents:
        context_text = "\n\n".join([f"[문서 {i+1}]\n{doc}" for i, doc in enumerate(documents)])
        thought_process.append(f"📝 금융 특화 답변 생성: {len(documents)}개 문서 기반")
    else:
        context_text = "관련 정보를 찾을 수 없습니다."
        thought_process.append("📝 금융 특화 답변 생성: 문서 없음 (일반 응답)")
    
    # 이전 대화 맥락 구성
    conversation_context = ""
    if messages and len(messages) > 0:
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        conversation_parts = []
        for msg in recent_messages:
            if hasattr(msg, 'content'):
                role = "사용자" if hasattr(msg, '__class__') and "Human" in msg.__class__.__name__ else "어시스턴트"
                conversation_parts.append(f"[{role}]: {msg.content}")
        if conversation_parts:
            conversation_context = "\n\n[이전 대화 맥락]\n" + "\n".join(conversation_parts) + "\n\n"
    
    # 금융 도메인별 특화 지시사항
    domain_instructions = {
        'stock': "주식 관련 정보를 제공할 때는 주가, 시가총액, PER, PBR, 기업 실적 등을 정확히 표시하세요.",
        'bond': "채권 관련 정보를 제공할 때는 수익률, 만기, 신용등급, 이자율 등을 명확히 구분하세요.",
        'forex': "외환 관련 정보를 제공할 때는 환율, 통화정책, 환차익 등을 시점 정보와 함께 제공하세요.",
        'real_estate': "부동산 관련 정보를 제공할 때는 지역, 시기, 가격 추이 등을 정확히 명시하세요.",
        'interest_rate': "금리 관련 정보를 제공할 때는 기준금리, 시장금리, 정책 금리 등을 구분하여 설명하세요.",
        'derivative': "파생상품 관련 정보를 제공할 때는 리스크를 명확히 경고하고 복잡성을 설명하세요.",
        'crypto': "암호화폐 관련 정보를 제공할 때는 변동성과 리스크를 강조하고 투자 조언이 아님을 명시하세요.",
        'economic': "경제 지표 관련 정보를 제공할 때는 데이터 출처, 시점, 단위를 정확히 표시하세요.",
        'general': "금융 정보를 제공할 때는 정확성과 신뢰성을 최우선으로 하세요."
    }
    domain_instruction = domain_instructions.get(financial_domain, domain_instructions['general'])
    
    # 신뢰도 기반 불확실성 표시
    confidence_note = ""
    if confidence_score < 0.7:
        confidence_note = f"\n[중요: 신뢰도 주의]\n이 답변의 신뢰도는 {confidence_score:.2f}로 비교적 낮습니다. 소스 간 일치도가 {source_agreement}이며, 추가 확인을 권장합니다."
    elif confidence_score >= 0.9:
        confidence_note = "\n[신뢰도: 높음]\n여러 신뢰할 수 있는 소스에서 일치하는 정보입니다."
    
    # 회사 비교 분석 정보 구성
    comparison_context = ""
    if company_comparison_data and financial_domain == "stock":
        target = company_comparison_data.get("target_company", "")
        industry = company_comparison_data.get("industry", "")
        market_cap = company_comparison_data.get("market_cap_category", "")
        similar = company_comparison_data.get("similar_companies", [])
        insights = company_comparison_data.get("comparison_insights", "")
        
        if target and industry and insights:
            comparison_context = f"""

[비교 분석 정보]
대상 회사: {target}
업종: {industry}
시가총액 규모: {market_cap}
비슷한 업종/규모의 회사: {', '.join(similar) if similar else '정보 없음'}
비교 인사이트: {insights}

위 비교 분석 정보를 참고하여 답변에 다음을 포함하세요:
- 비슷한 업종/규모의 회사들과의 비교 (주가, 실적, 밸류에이션 지표 등)
- 업종 평균 대비 위치 분석
- 비교를 통한 추론 및 인사이트
"""
    
    # 금융 특화 프롬프트
    prompt = f"""당신은 전문 금융 분석가입니다. 다음 질문에 정확하고 신뢰할 수 있는 답변을 제공하세요.

[금융 도메인] {financial_domain}
[현재 시점] {full_datetime_str}
[신뢰도] {confidence_score:.2f} (소스 일치도: {source_agreement})
{confidence_note}

[금융 답변 가이드라인]
1. 정확성: 모든 숫자, 비율, 통계, 날짜를 정확히 기재
2. 출처 명시: 정보 출처를 가능한 한 명시
3. 불확실성 표시: 불확실하거나 한계가 있는 정보는 명확히 표시
4. 최신성: 시점 정보 포함 (예: "2024년 12월 기준")
5. 리스크 경고: 투자 관련 질문은 반드시 리스크 경고 포함
6. 법적 면책: 투자 조언이 아님을 명시 ("본 답변은 정보 제공 목적이며, 투자 조언이 아닙니다.")
7. 도메인 특화: {domain_instruction}

{conversation_context}
[검색된 문서 컨텍스트]
{context_text}
{comparison_context}
[현재 질문]
{question}

[답변] (위 가이드라인을 모두 준수하여 답변하세요. 비교 분석 정보가 있으면 반드시 포함하여 더 풍부한 인사이트를 제공하세요.)
"""
    
    try:
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        thought_process.append("✅ 금융 특화 답변 생성 완료")
    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg or "429" in error_msg:
            answer = "⚠️ OpenAI API 할당량 오류가 발생했습니다. 계정 크레딧을 확인하세요."
        else:
            answer = f"죄송합니다. 답변을 생성하는 중 오류가 발생했습니다.\n\n오류 내용: {error_msg}"
    
    updated_messages = messages + [
        HumanMessage(content=question),
        AIMessage(content=answer)
    ]
    
    return {
        "messages": updated_messages,
        "context": context_text,
        "current_step": "완료",
        "thought_process": thought_process
    }


def extract_and_compare_companies(state: AgentState) -> AgentState:
    """
    회사 정보 추출 및 비교 분석 노드
    질문에서 회사명을 추출하고, 비슷한 업종/규모의 회사와 비교 분석합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 (company_comparison_data 포함)
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    financial_domain = state.get("financial_domain", "general")
    thought_process = state.get("thought_process", [])
    
    # 주식 도메인이 아니면 스킵
    if financial_domain != "stock":
        thought_process.append("🔍 회사 비교 분석: 주식 도메인이 아니므로 스킵")
        return {
            "company_comparison_data": None,
            "thought_process": thought_process
        }
    
    # 질문에서 회사명 추출
    company_extraction_prompt = f"""다음 질문에서 회사명이나 기업명을 추출해주세요.

질문: {question}

회사명이나 기업명이 있으면 그 이름만 답변하고, 없으면 "없음"이라고 답변하세요.
답변 형식: 회사명만 (예: "삼성전자", "애플", "없음"): """
    
    try:
        response = llm.invoke(company_extraction_prompt)
        company_name = response.content.strip() if hasattr(response, 'content') else "없음"
        
        if "없음" in company_name or len(company_name) < 2:
            thought_process.append("🔍 회사 비교 분석: 회사명 추출 실패")
            return {
                "company_comparison_data": None,
                "thought_process": thought_process
            }
        
        thought_process.append(f"🔍 회사명 추출: {company_name}")
    except Exception:
        thought_process.append("🔍 회사 비교 분석: 회사명 추출 오류")
        return {
            "company_comparison_data": None,
            "thought_process": thought_process
        }
    
    # 문서에서 회사 정보 추출
    documents_text = "\n\n".join([f"[문서 {i+1}]\n{doc[:2000]}..." if len(doc) > 2000 else f"[문서 {i+1}]\n{doc}" 
                                  for i, doc in enumerate(documents)])
    
    # 회사 정보 및 비교 분석 요청
    comparison_prompt = f"""다음 정보를 바탕으로 {company_name} 회사에 대한 비교 분석을 수행해주세요.

[검색된 문서]
{documents_text[:5000]}  # 문서가 너무 길면 잘라내기

[분석 요청사항]
1. {company_name}의 업종(산업 분야)을 파악하세요
2. {company_name}의 시가총액 규모를 파악하세요 (대형/중형/소형)
3. 같은 업종에서 비슷한 규모의 경쟁사나 비교 가능한 회사 3-5개를 제시하세요
4. 해당 회사들과의 비교 인사이트를 제공하세요 (주가, 실적, PER, PBR 등)

답변을 JSON 형식으로 제공하세요:
{{
    "target_company": "{company_name}",
    "industry": "업종명",
    "market_cap_category": "대형/중형/소형",
    "similar_companies": ["회사1", "회사2", "회사3"],
    "comparison_insights": "비교 분석 인사이트 (주가, 실적, 밸류에이션 등 비교)"
}}

정보가 부족하면 가능한 부분만 답변하세요: """
    
    try:
        response = llm.invoke(comparison_prompt)
        comparison_text = response.content.strip() if hasattr(response, 'content') else ""
        
        # JSON 추출 시도 (간단한 파싱)
        import json
        import re
        
        # JSON 블록 찾기
        json_match = re.search(r'\{.*\}', comparison_text, re.DOTALL)
        if json_match:
            try:
                comparison_data = json.loads(json_match.group())
                thought_process.append(f"✅ 회사 비교 분석 완료: {company_name} ({comparison_data.get('industry', 'N/A')} 업종)")
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 텍스트에서 정보 추출
                comparison_data = {
                    "target_company": company_name,
                    "industry": "정보 부족",
                    "market_cap_category": "정보 부족",
                    "similar_companies": [],
                    "comparison_insights": comparison_text[:500] if comparison_text else "비교 분석 정보를 추출할 수 없습니다."
                }
                thought_process.append(f"⚠️ 회사 비교 분석: JSON 파싱 실패, 텍스트 정보 사용")
        else:
            # JSON이 없으면 텍스트 정보만 사용
            comparison_data = {
                "target_company": company_name,
                "industry": "정보 부족",
                "market_cap_category": "정보 부족",
                "similar_companies": [],
                "comparison_insights": comparison_text[:500] if comparison_text else "비교 분석 정보를 추출할 수 없습니다."
            }
            thought_process.append(f"⚠️ 회사 비교 분석: JSON 형식 없음, 텍스트 정보 사용")
        
    except Exception as e:
        comparison_data = {
            "target_company": company_name,
            "industry": "오류",
            "market_cap_category": "오류",
            "similar_companies": [],
            "comparison_insights": f"비교 분석 중 오류 발생: {str(e)}"
        }
        thought_process.append(f"⚠️ 회사 비교 분석: 오류 발생")
    
    return {
        "company_comparison_data": comparison_data,
        "thought_process": thought_process
    }


def calculate_confidence(state: AgentState) -> AgentState:
    """
    신뢰도 계산 노드: 여러 요소를 종합하여 최종 신뢰도 계산
    금융 정보의 신뢰성을 종합적으로 평가합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 (confidence_score 포함)
    """
    documents = state.get("documents", [])
    is_relevant = state.get("is_relevant", "no")
    source_agreement = state.get("source_agreement", "low")
    verification_round = state.get("verification_round", 0)
    loop_count = state.get("loop_count", 0)
    thought_process = state.get("thought_process", [])
    
    # 신뢰도 계산 로직
    base_confidence = 0.4  # 기본 신뢰도
    
    # 관련성 점수 (0.3)
    if is_relevant == "yes":
        base_confidence += 0.3
    elif is_relevant == "partial":
        base_confidence += 0.15
    
    # 소스 일치도 점수 (0.2)
    if source_agreement == "high":
        base_confidence += 0.2
    elif source_agreement == "medium":
        base_confidence += 0.1
    
    # 검증 라운드 수 (0.1) - 더 많이 검증할수록 신뢰도 증가
    base_confidence += min(verification_round * 0.05, 0.1)
    
    # 문서 수 (0.1) - 충분한 소스가 있을수록 신뢰도 증가
    if len(documents) >= 5:
        base_confidence += 0.1
    elif len(documents) >= 3:
        base_confidence += 0.07
    elif len(documents) >= 2:
        base_confidence += 0.03
    
    # 너무 많은 루프는 오히려 신뢰도 감소 (0.1 패널티)
    if loop_count > 5:
        base_confidence -= 0.1
    
    confidence_score = max(0.0, min(base_confidence, 1.0))
    
    thought_process.append(f"📊 신뢰도 계산: {confidence_score:.2f} (관련성: {is_relevant}, 소스일치: {source_agreement}, 검증: {verification_round}라운드)")
    
    return {
        "confidence_score": confidence_score,
        "thought_process": thought_process
    }


def load_pdf_to_vectorstore(file_path: str, persist_directory: str = "./chroma_db"):
    """
    PDF 파일을 업로드하여 텍스트를 쪼개고 ChromaDB에 저장한 후 Retriever를 반환하는 함수
    
    Args:
        file_path: PDF 파일 경로
        persist_directory: ChromaDB 저장 디렉토리
        
    Returns:
        Retriever 객체
    """
    # PDF 로드
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 텍스트 분할 (Split)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # ChromaDB 벡터 스토어에 저장
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Retriever 생성 및 반환
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    print(f"문서가 벡터 스토어에 저장되었습니다: {len(splits)}개 청크")
    return retriever

