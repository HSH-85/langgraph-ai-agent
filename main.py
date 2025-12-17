"""
LangGraph RAG 에이전트의 메인 그래프 정의 (고도화)
"""
import sys
import io

# UTF-8 인코딩 설정 (Windows 콘솔 호환)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from langgraph.graph import StateGraph, START, END
from nodes import analyze_intent, retrieve, rerank_documents, grade_documents, web_search_node, generate
from state import AgentState


def decide_to_search(state: AgentState) -> str:
    """
    웹 검색이 필요한지 판단하는 조건부 함수 (고도화)
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        "web_search" 또는 "generate"
    """
    web_search = state.get("web_search", False)
    loop_count = state.get("loop_count", 0)
    
    # 최대 재시도 횟수 초과 시 강제로 generate로 이동
    if loop_count >= 3:
        return "generate"
    
    if web_search:
        return "web_search"
    else:
        return "generate"


def create_agent_graph():
    """
    고도화된 RAG 에이전트 그래프 생성 및 컴파일
    
    워크플로우:
    1. analyze_intent: 사용자 질문 의도 분석
    2. retrieve: 벡터 스토어에서 문서 검색
    3. rerank: Cohere로 문서 재정렬
    4. grade_documents: LLM으로 문서 관련성 평가
    5. web_search (조건부): 관련성 낮으면 웹 검색
    6. generate: 최종 답변 생성
    
    Returns:
        컴파일된 LangGraph 그래프 (app)
    """
    # 그래프 생성
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("analyze_intent", analyze_intent)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", rerank_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("generate", generate)
    
    # 엣지 연결
    # 시작 -> 의도 분석
    workflow.add_edge(START, "analyze_intent")
    
    # 의도 분석 -> 문서 검색
    workflow.add_edge("analyze_intent", "retrieve")
    
    # retrieve -> rerank (리랭크)
    workflow.add_edge("retrieve", "rerank")
    
    # rerank -> grade_documents
    workflow.add_edge("rerank", "grade_documents")
    
    # grade_documents -> (조건부 엣지)
    # web_search가 True면 -> web_search_node, 아니면 -> generate
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_search,
        {
            "web_search": "web_search_node",
            "generate": "generate"
        }
    )
    
    # web_search_node -> generate
    workflow.add_edge("web_search_node", "generate")
    
    # generate -> 종료(END)
    workflow.add_edge("generate", END)
    
    # 그래프 컴파일
    app = workflow.compile()
    
    return app


def run_agent(question: str, graph=None):
    """
    고도화된 에이전트 실행 함수
    
    Args:
        question: 사용자 질문
        graph: 컴파일된 그래프 (None이면 새로 생성)
        
    Returns:
        최종 상태 딕셔너리 (messages, thought_process, context 등 포함)
    """
    if graph is None:
        graph = create_agent_graph()
    
    # 고도화된 초기 상태 설정
    initial_state = {
        "messages": [],
        "question": question,
        "intent": None,
        "documents": [],
        "context": None,
        "is_relevant": None,
        "web_search": False,
        "loop_count": 0,
        "current_step": "의도 분석 중...",
        "thought_process": []
    }
    
    # 그래프 실행
    result = graph.invoke(initial_state)
    
    return result


if __name__ == "__main__":
    # 테스트 실행
    print("=" * 80)
    print("고도화된 LangGraph RAG 에이전트 테스트")
    print("=" * 80)
    
    graph = create_agent_graph()
    
    test_question = "LangGraph란 무엇인가요?"
    print(f"\n질문: {test_question}\n")
    
    result = run_agent(test_question, graph)
    
    # 사고 과정 출력
    print("\n" + "=" * 80)
    print("[사고 과정]")
    print("=" * 80)
    for thought in result.get("thought_process", []):
        print(f"  {thought}")
    
    # 최종 답변 출력
    print("\n" + "=" * 80)
    print("[최종 답변]")
    print("=" * 80)
    if result.get("messages"):
        last_message = result["messages"][-1]
        if hasattr(last_message, 'content'):
            print(last_message.content)
        else:
            print(last_message)
    
    # 메타 정보 출력
    print("\n" + "=" * 80)
    print("[메타 정보]")
    print("=" * 80)
    print(f"  의도: {result.get('intent', 'N/A')}")
    print(f"  관련성: {result.get('is_relevant', 'N/A')}")
    print(f"  웹 검색 사용: {'예' if result.get('web_search') else '아니오'}")
    print(f"  재시도 횟수: {result.get('loop_count', 0)}")
    print(f"  문서 개수: {len(result.get('documents', []))}")
    print("=" * 80)

