"""
LangGraph RAG 에이전트의 메인 그래프 정의
"""
from langgraph.graph import StateGraph, START, END
from nodes import retrieve, grade_documents, web_search_node, generate
from state import GraphState


def decide_to_search(state: GraphState) -> str:
    """
    웹 검색이 필요한지 판단하는 조건부 함수
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        "web_search" 또는 "generate"
    """
    web_search = state.get("web_search", False)
    
    if web_search:
        return "web_search"
    else:
        return "generate"


def create_agent_graph():
    """
    RAG 에이전트 그래프 생성 및 컴파일
    
    Returns:
        컴파일된 LangGraph 그래프 (app)
    """
    # 그래프 생성
    workflow = StateGraph(GraphState)
    
    # 노드 추가
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("generate", generate)
    
    # 엣지 연결
    # 시작 -> retrieve
    workflow.add_edge(START, "retrieve")
    
    # retrieve -> grade_documents
    workflow.add_edge("retrieve", "grade_documents")
    
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
    에이전트 실행 함수
    
    Args:
        question: 사용자 질문
        graph: 컴파일된 그래프 (None이면 새로 생성)
        
    Returns:
        최종 상태 딕셔너리
    """
    if graph is None:
        graph = create_agent_graph()
    
    # 초기 상태 설정
    initial_state = {
        "messages": [],
        "question": question,
        "documents": [],
        "web_search": False
    }
    
    # 그래프 실행
    result = graph.invoke(initial_state)
    
    return result


if __name__ == "__main__":
    # 테스트 실행
    graph = create_agent_graph()
    
    test_question = "LangGraph란 무엇인가요?"
    print(f"질문: {test_question}\n")
    
    result = run_agent(test_question, graph)
    
    # 최종 답변만 깔끔하게 출력
    if result.get("messages"):
        last_message = result["messages"][-1]
        if hasattr(last_message, 'content'):
            print("=" * 60)
            print("답변:")
            print("=" * 60)
            print(last_message.content)
        else:
            print("=" * 60)
            print("답변:")
            print("=" * 60)
            print(last_message)

