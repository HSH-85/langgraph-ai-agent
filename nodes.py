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
    
    # 무한 루프 방지: 최대 3회까지만 재시도
    if loop_count >= 3:
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
        
        # 검색 쿼리에 현재 시간 정보 추가 (최신 정보 검색을 위해)
        enhanced_query = f"{question} (현재 시간: {current_datetime_str})"
        
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
                timestamped_content = f"[검색 시간: {current_datetime_str}]\n{content}"
                search_results.append(timestamped_content)
        
        existing_documents.extend(search_results)
        thought_process.append(f"🌐 웹 검색: {len(search_results)}개 결과 추가 (검색 시간: {current_datetime_str})")
    except Exception as e:
        # 웹 검색 오류 시 기존 문서 유지 (조용히 처리)
        thought_process.append("⚠️ 웹 검색: 오류 발생")
    
    return {
        "documents": existing_documents,
        "web_search": False,  # 웹 검색 완료 후 플래그 리셋
        "current_step": "답변 생성 중...",
        "thought_process": thought_process
    }


def generate(state: AgentState) -> AgentState:
    """
    확보된 문서들을 context로 삼아 최종 답변을 생성하는 노드 (고도화)
    현재 시간 정보를 컨텍스트에 포함하여 정확한 시간 정보를 제공합니다.
    
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
    
    # 프롬프트 구성 (현재 시간 정보 포함)
    prompt = f"""다음 컨텍스트를 기반으로 질문에 답변해주세요.

[중요: 현재 시간 정보]
현재 시간은 {full_datetime_str}입니다.
시간 관련 질문이 있다면 이 정보를 기준으로 답변하세요.

[답변 스타일]
{intent_instruction}

[컨텍스트]
{context_text}

[질문]
{question}

[답변]
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

