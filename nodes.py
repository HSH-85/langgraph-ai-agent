"""
LangGraph RAG 에이전트의 노드 함수들
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage
from tavily import TavilyClient
from state import GraphState
import os
from dotenv import load_dotenv

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


# 전역 변수: Retriever (초기화 필요)
retriever = None


def set_retriever(new_retriever):
    """Retriever를 설정하는 함수"""
    global retriever
    retriever = new_retriever


def retrieve(state: GraphState) -> GraphState:
    """
    질문을 받아 VectorDB에서 문서를 검색하는 노드
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 (documents 포함)
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    
    # Retriever가 설정되지 않은 경우 ChromaDB에서 직접 검색
    if retriever is None:
        try:
            vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
            docs = vectorstore.similarity_search(question, k=3)
            documents = [doc.page_content for doc in docs]
        except Exception:
            # 벡터 스토어가 없거나 오류 발생 시 빈 리스트 반환 (조용히 처리)
            documents = []
    else:
        # Retriever를 사용하여 문서 검색
        try:
            docs = retriever.invoke(question)
            documents = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs]
        except Exception:
            # Retriever 검색 오류 시 빈 리스트 반환 (조용히 처리)
            documents = []
    
    return {
        "documents": documents
    }


def grade_documents(state: GraphState) -> GraphState:
    """
    검색된 문서가 질문과 관련 있는지 LLM(gpt-4o-mini)으로 평가하는 노드
    관련 없으면 web_search=True 설정
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 (web_search 포함)
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    
    # 문서가 없으면 웹 검색 필요
    if not documents or len(documents) == 0:
        return {
            "web_search": True
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
    except Exception as e:
        # 오류 발생 시 웹 검색으로 폴백 (조용히 처리)
        web_search_needed = True
    
    return {
        "web_search": web_search_needed
    }


def web_search_node(state: GraphState) -> GraphState:
    """
    web_search=True일 때 Tavily API로 웹 검색을 수행하고 결과를 문서에 추가하는 노드
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 (documents에 웹 검색 결과 추가)
    """
    question = state.get("question", "")
    existing_documents = state.get("documents", [])
    
    if tavily_client is None:
        print("⚠️ Tavily 클라이언트가 초기화되지 않았습니다. 웹 검색을 건너뜁니다.")
        return {
            "documents": existing_documents,
            "web_search": False
        }
    
    try:
        # Tavily 검색
        response = tavily_client.search(
            query=question,
            max_results=3,
            search_depth="advanced"
        )
        
        # 검색 결과를 문서에 추가
        search_results = [
            result.get("content", "") 
            for result in response.get("results", [])
            if result.get("content")
        ]
        existing_documents.extend(search_results)
    except Exception as e:
        # 웹 검색 오류 시 기존 문서 유지 (조용히 처리)
        pass
    
    return {
        "documents": existing_documents,
        "web_search": False  # 웹 검색 완료 후 플래그 리셋
    }


def generate(state: GraphState) -> GraphState:
    """
    확보된 문서들을 context로 삼아 최종 답변을 생성하는 노드
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 (messages에 답변 추가)
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    messages = state.get("messages", [])
    
    # 문서들을 컨텍스트로 결합
    if documents:
        context_text = "\n\n".join([f"[문서 {i+1}]\n{doc}" for i, doc in enumerate(documents)])
    else:
        context_text = "관련 정보를 찾을 수 없습니다."
    
    # 프롬프트 구성
    prompt = f"""다음 컨텍스트를 기반으로 질문에 정확하고 상세하게 답변해주세요.
컨텍스트에 없는 정보는 추측하지 말고, 컨텍스트에 있는 정보만 사용하여 답변해주세요.

컨텍스트:
{context_text}

질문: {question}

답변:"""

    # LLM을 사용하여 답변 생성
    try:
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
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
        "messages": updated_messages
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

