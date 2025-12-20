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
from nodes import (
    analyze_intent, retrieve, rerank_documents, grade_documents, web_search_node, generate,
    analyze_financial_domain, verify_documents, cross_validate, calculate_confidence, generate_financial,
    extract_and_compare_companies
)
from state import AgentState


def decide_next_step(state: AgentState) -> str:
    """
    금융 특화: 다음 단계 결정 함수
    다중 검색, 검증 라운드를 고려하여 다음 단계를 결정합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        다음 노드 이름 ("web_search_node", "verify_documents", "cross_validate", "calculate_confidence", "generate")
    """
    web_search = state.get("web_search", False)
    loop_count = state.get("loop_count", 0)
    search_round = state.get("search_round", 0)
    verification_round = state.get("verification_round", 0)
    confidence_score = state.get("confidence_score")
    
    MAX_LOOPS = 7
    MAX_SEARCH_ROUNDS = 3
    MAX_VERIFICATION_ROUNDS = 2
    MIN_CONFIDENCE = 0.7
    
    # 최대 루프 초과 시 강제로 신뢰도 계산 후 생성
    if loop_count >= MAX_LOOPS:
        if confidence_score is None:
            return "calculate_confidence"
        return "generate"
    
    # 웹 검색이 필요한 경우
    if web_search:
        # 검색 라운드가 남아있으면 웹 검색
        if search_round < MAX_SEARCH_ROUNDS:
            return "web_search_node"
        # 검색 라운드 완료 후 크로스 검증
        else:
            return "cross_validate"
    
    # 웹 검색이 필요 없는 경우
    # 신뢰도가 아직 계산되지 않았으면 계산
    if confidence_score is None:
        return "calculate_confidence"
    
    # 신뢰도가 낮고 검증 라운드가 남아있으면 검증
    if confidence_score < MIN_CONFIDENCE and verification_round < MAX_VERIFICATION_ROUNDS:
        return "verify_documents"
    
    # 신뢰도가 충분하거나 모든 검증이 완료되면 답변 생성
    return "generate"


# decide_based_on_confidence 함수는 더 이상 사용하지 않음 (calculate_confidence 후 바로 generate로 이동)


def create_agent_graph():
    """
    금융 특화 고도화된 RAG 에이전트 그래프 생성 및 컴파일
    
    워크플로우:
    1. analyze_intent: 사용자 질문 의도 분석
    2. analyze_financial_domain: 금융 도메인 분류
    3. retrieve: 벡터 스토어에서 문서 검색
    4. rerank: Cohere로 문서 재정렬
    5. grade_documents: LLM으로 문서 관련성 평가
    6. verify_documents (조건부): 추가 검증 라운드
    7. web_search (조건부): 관련성 낮으면 웹 검색 (다중 라운드)
    8. cross_validate: 크로스 검증
    9. calculate_confidence: 신뢰도 계산
    10. generate_financial: 금융 특화 답변 생성
    
    Returns:
        컴파일된 LangGraph 그래프 (app)
    """
    # 그래프 생성
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("analyze_intent", analyze_intent)
    workflow.add_node("analyze_financial_domain", analyze_financial_domain)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", rerank_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("verify_documents", verify_documents)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("cross_validate", cross_validate)
    workflow.add_node("calculate_confidence", calculate_confidence)
    workflow.add_node("extract_and_compare_companies", extract_and_compare_companies)
    workflow.add_node("generate", generate_financial)
    
    # 엣지 연결
    # 시작 -> 의도 분석
    workflow.add_edge(START, "analyze_intent")
    
    # 의도 분석 -> 금융 도메인 분석
    workflow.add_edge("analyze_intent", "analyze_financial_domain")
    
    # 금융 도메인 분석 -> 문서 검색
    workflow.add_edge("analyze_financial_domain", "retrieve")
    
    # retrieve -> rerank (리랭크)
    workflow.add_edge("retrieve", "rerank")
    
    # rerank -> grade_documents
    workflow.add_edge("rerank", "grade_documents")
    
    # grade_documents -> (조건부 엣지: 다중 검색/검증 라운드 지원)
    workflow.add_conditional_edges(
        "grade_documents",
        decide_next_step,
        {
            "web_search_node": "web_search_node",
            "verify_documents": "verify_documents",
            "cross_validate": "cross_validate",
            "calculate_confidence": "calculate_confidence",
            "generate": "generate"
        }
    )
    
    # verify_documents -> grade_documents (재평가)
    workflow.add_edge("verify_documents", "grade_documents")
    
    # web_search_node -> cross_validate
    workflow.add_edge("web_search_node", "cross_validate")
    
    # cross_validate -> calculate_confidence
    workflow.add_edge("cross_validate", "calculate_confidence")
    
    # calculate_confidence -> extract_and_compare_companies (주식 도메인인 경우)
    workflow.add_edge("calculate_confidence", "extract_and_compare_companies")
    
    # extract_and_compare_companies -> generate
    workflow.add_edge("extract_and_compare_companies", "generate")
    
    # generate -> 종료(END)
    workflow.add_edge("generate", END)
    
    # 그래프 컴파일 (재귀 제한 설정)
    app = workflow.compile()
    
    return app


def run_agent(question: str, graph=None, previous_messages=None):
    """
    고도화된 에이전트 실행 함수
    
    Args:
        question: 사용자 질문
        graph: 컴파일된 그래프 (None이면 새로 생성)
        previous_messages: 이전 대화 메시지 리스트 (대화 맥락 유지용)
        
    Returns:
        최종 상태 딕셔너리 (messages, thought_process, context 등 포함)
    """
    if graph is None:
        graph = create_agent_graph()
    
    # 이전 메시지가 없으면 빈 리스트로 초기화
    if previous_messages is None:
        previous_messages = []
    
    # 현재 질문을 HumanMessage로 추가
    from langchain_core.messages import HumanMessage
    current_message = HumanMessage(content=question)
    
    # 고도화된 초기 상태 설정 (이전 메시지 포함, 금융 특화 필드 포함)
    # messages 필드는 LangGraph가 자동으로 누적하므로, 이전 메시지만 포함
    initial_state = {
        "messages": previous_messages + [current_message],  # 이전 대화 기록 + 현재 질문
        "question": question,
        "intent": None,
        "documents": [],
        "context": None,
        "is_relevant": None,
        "web_search": False,
        "loop_count": 0,
        # 금융 특화 필드
        "search_round": 0,
        "verification_round": 0,
        "financial_domain": None,
        "confidence_score": None,
        "source_agreement": None,
        "cross_validation_results": [],
        "additional_search_queries": [],
        "company_comparison_data": None,
        # UI/UX
        "current_step": "의도 분석 중...",
        "thought_process": []
    }
    
    # 그래프 실행 (재귀 제한 설정: 최대 50회)
    config = {"recursion_limit": 50}
    try:
        result = graph.invoke(initial_state, config=config)
    except Exception as e:
        # 에러 발생 시 상세 정보 출력
        import traceback
        print(f"에러 발생: {e}")
        print(f"에러 타입: {type(e)}")
        traceback.print_exc()
        raise
    
    return result


if __name__ == "__main__":
    # 테스트 실행
    print("=" * 80)
    print("고도화된 LangGraph RAG 에이전트 테스트")
    print("=" * 80)
    
    graph = create_agent_graph()
    
    test_question = "삼성전자 주가는 얼마인가요?"
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

