"""
고도화 전후 버전 비교 스크립트
동일한 질문에 대해 두 버전을 실행하고 정확도, 신뢰도, 성능 등을 비교합니다.
"""
import sys
import io
import time
from datetime import datetime
from typing import Dict, Any

# UTF-8 인코딩 설정 (Windows 콘솔 호환) - main.py에서 이미 설정했을 수 있으므로 try-except로 처리
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer') and not isinstance(sys.stderr, io.TextIOWrapper):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, ValueError, OSError):
        # 이미 래핑되었거나 버퍼가 없는 경우 무시
        pass

# main.py import는 함수 내부로 이동하여 인코딩 설정 후 import


def create_simple_graph():
    """
    간단한 버전의 그래프 (고도화 전 버전 시뮬레이션)
    금융 특화 기능 없이 기본 워크플로우만 실행
    """
    # import를 함수 내부로 이동
    from langgraph.graph import StateGraph, START, END
    import sys
    import os
    # 현재 파일의 디렉토리를 sys.path에 추가
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from nodes import analyze_intent, retrieve, rerank_documents, grade_documents, web_search_node, generate
    from state import AgentState
    
    def simple_decide_to_search(state: AgentState) -> str:
        web_search = state.get("web_search", False)
        loop_count = state.get("loop_count", 0)
        if loop_count >= 3:
            return "generate"
        return "web_search" if web_search else "generate"
    
    workflow = StateGraph(AgentState)
    workflow.add_node("analyze_intent", analyze_intent)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", rerank_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("generate", generate)
    
    workflow.add_edge(START, "analyze_intent")
    workflow.add_edge("analyze_intent", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        simple_decide_to_search,
        {"web_search": "web_search_node", "generate": "generate"}
    )
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


def run_simple_agent(question: str, graph, previous_messages=None):
    """간단한 버전 실행"""
    if previous_messages is None:
        previous_messages = []
    
    # 기본 필드만 포함 (금융 특화 필드 제외)
    initial_state = {
        "messages": previous_messages,
        "question": question,
        "intent": None,
        "documents": [],
        "context": None,
        "is_relevant": None,
        "web_search": False,
        "loop_count": 0,
        # 금융 특화 필드는 기본값으로 설정 (State에 필수 필드로 있으므로)
        "search_round": 0,
        "verification_round": 0,
        "financial_domain": None,
        "confidence_score": None,
        "source_agreement": None,
        "cross_validation_results": [],
        "additional_search_queries": [],
        "current_step": "의도 분석 중...",
        "thought_process": []
    }
    
    return graph.invoke(initial_state)


def extract_metrics(result: Dict[str, Any], execution_time: float = None) -> Dict[str, Any]:
    """결과에서 메트릭 추출 (성능 메트릭 포함)"""
    answer_content = ""
    if result.get("messages") and len(result.get("messages", [])) > 0:
        last_msg = result["messages"][-1]
        if hasattr(last_msg, 'content'):
            answer_content = last_msg.content
        else:
            answer_content = str(last_msg)
    
    # 사고 과정에서 LLM 호출 횟수 추정 (의도 분석, 평가, 생성 등)
    thought_process = result.get("thought_process", [])
    llm_call_indicators = ["의도 분석", "문서 평가", "답변 생성", "금융 도메인 분석", "문서 검증", "크로스 검증", "신뢰도 계산"]
    estimated_llm_calls = sum(1 for thought in thought_process if any(indicator in thought for indicator in llm_call_indicators))
    
    return {
        "answer_length": len(answer_content),
        "doc_count": len(result.get("documents", [])),
        "loop_count": result.get("loop_count", 0),
        "thought_process_count": len(thought_process),
        "is_relevant": result.get("is_relevant"),
        "intent": result.get("intent"),
        # 금융 특화 메트릭 (없으면 None)
        "financial_domain": result.get("financial_domain"),
        "confidence_score": result.get("confidence_score"),
        "source_agreement": result.get("source_agreement"),
        "search_round": result.get("search_round", 0),
        "verification_round": result.get("verification_round", 0),
        # 성능 메트릭
        "execution_time": execution_time,
        "estimated_llm_calls": estimated_llm_calls,
    }


def compare_results(question: str, simple_result: Dict, advanced_result: Dict, 
                    simple_time: float = None, advanced_time: float = None):
    """두 결과를 비교하여 리포트 생성 (성능 메트릭 포함)"""
    simple_metrics = extract_metrics(simple_result, simple_time)
    advanced_metrics = extract_metrics(advanced_result, advanced_time)
    
    print("\n" + "=" * 80)
    print("📊 비교 결과 리포트")
    print("=" * 80)
    print(f"\n질문: {question}\n")
    
    print("-" * 80)
    print("🔹 기본 메트릭 비교")
    print("-" * 80)
    print(f"{'항목':<30} {'기본 버전':<25} {'금융 특화 버전':<25}")
    print("-" * 80)
    print(f"{'문서 개수':<30} {simple_metrics['doc_count']:<25} {advanced_metrics['doc_count']:<25}")
    print(f"{'루프 횟수':<30} {simple_metrics['loop_count']:<25} {advanced_metrics['loop_count']:<25}")
    print(f"{'의도 분석':<30} {simple_metrics['intent'] or 'N/A':<25} {advanced_metrics['intent'] or 'N/A':<25}")
    print(f"{'관련성 평가':<30} {simple_metrics['is_relevant'] or 'N/A':<25} {advanced_metrics['is_relevant'] or 'N/A':<25}")
    print(f"{'답변 길이 (자)':<30} {simple_metrics['answer_length']:<25} {advanced_metrics['answer_length']:<25}")
    print(f"{'사고 과정 단계 수':<30} {simple_metrics['thought_process_count']:<25} {advanced_metrics['thought_process_count']:<25}")
    
    # 성능 메트릭 표시
    if simple_metrics.get('execution_time') is not None and advanced_metrics.get('execution_time') is not None:
        simple_time_str = f"{simple_metrics['execution_time']:.2f}초"
        advanced_time_str = f"{advanced_metrics['execution_time']:.2f}초"
        time_diff = advanced_metrics['execution_time'] - simple_metrics['execution_time']
        time_diff_str = f"{time_diff:+.2f}초" if abs(time_diff) >= 0.1 else "거의 동일"
        print(f"{'실행 시간':<30} {simple_time_str:<25} {advanced_time_str} ({time_diff_str})")
    
    if simple_metrics.get('estimated_llm_calls') is not None and advanced_metrics.get('estimated_llm_calls') is not None:
        print(f"{'예상 LLM 호출 횟수':<30} {simple_metrics['estimated_llm_calls']:<25} {advanced_metrics['estimated_llm_calls']:<25}")
    
    print("\n" + "-" * 80)
    print("🔹 금융 특화 메트릭 (기본 버전에는 없음)")
    print("-" * 80)
    print(f"{'금융 도메인':<30} {'N/A (미지원)':<25} {advanced_metrics['financial_domain'] or 'N/A':<25}")
    print(f"{'신뢰도 점수':<30} {'N/A (미지원)':<25} {advanced_metrics['confidence_score'] or 'N/A':<25}")
    print(f"{'소스 일치도':<30} {'N/A (미지원)':<25} {advanced_metrics['source_agreement'] or 'N/A':<25}")
    print(f"{'검색 라운드':<30} {'N/A (미지원)':<25} {advanced_metrics['search_round']:<25}")
    print(f"{'검증 라운드':<30} {'N/A (미지원)':<25} {advanced_metrics['verification_round']:<25}")
    
    print("\n" + "-" * 80)
    print("🔹 개선 사항 분석")
    print("-" * 80)
    
    improvements = []
    if advanced_metrics['doc_count'] > simple_metrics['doc_count']:
        improvements.append(f"✅ 문서 개수 증가: {simple_metrics['doc_count']} → {advanced_metrics['doc_count']} (+{advanced_metrics['doc_count'] - simple_metrics['doc_count']})")
    
    if advanced_metrics['thought_process_count'] > simple_metrics['thought_process_count']:
        improvements.append(f"✅ 검증 단계 증가: {simple_metrics['thought_process_count']} → {advanced_metrics['thought_process_count']} 단계")
    
    if advanced_metrics['confidence_score'] is not None:
        if advanced_metrics['confidence_score'] >= 0.7:
            improvements.append(f"✅ 높은 신뢰도: {advanced_metrics['confidence_score']:.2%}")
        else:
            improvements.append(f"⚠️ 신뢰도 개선 필요: {advanced_metrics['confidence_score']:.2%}")
    
    if advanced_metrics['source_agreement'] == 'high':
        improvements.append("✅ 소스 간 높은 일치도 확인")
    
    if advanced_metrics['search_round'] > 0:
        improvements.append(f"✅ 다중 검색 라운드 실행: {advanced_metrics['search_round']}라운드")
    
    if advanced_metrics['verification_round'] > 0:
        improvements.append(f"✅ 문서 검증 라운드 실행: {advanced_metrics['verification_round']}라운드")
    
    if not improvements:
        improvements.append("⚠️ 개선 사항이 명확하지 않습니다. 질문 유형에 따라 다를 수 있습니다.")
    
    for improvement in improvements:
        print(f"  {improvement}")
    
    print("\n" + "-" * 80)
    print("🔹 답변 미리보기")
    print("-" * 80)
    
    # 기본 버전 답변
    if simple_result.get("messages"):
        simple_answer = simple_result["messages"][-1].content if hasattr(simple_result["messages"][-1], 'content') else str(simple_result["messages"][-1])
        print(f"\n[기본 버전] (길이: {len(simple_answer)}자)")
        print(simple_answer[:300] + "..." if len(simple_answer) > 300 else simple_answer)
    
    # 금융 특화 버전 답변
    if advanced_result.get("messages"):
        advanced_answer = advanced_result["messages"][-1].content if hasattr(advanced_result["messages"][-1], 'content') else str(advanced_result["messages"][-1])
        print(f"\n[금융 특화 버전] (길이: {len(advanced_answer)}자)")
        print(advanced_answer[:300] + "..." if len(advanced_answer) > 300 else advanced_answer)
    
    print("\n" + "=" * 80)


def run_comparison(questions: list):
    """여러 질문에 대해 비교 실행"""
    # import를 함수 내부로 이동
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from main import create_agent_graph, run_agent
    
    print("=" * 80)
    print("🔬 고도화 전후 버전 비교 테스트")
    print("=" * 80)
    print(f"\n테스트 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"테스트 질문 수: {len(questions)}\n")
    
    # 그래프 생성
    print("그래프 생성 중...")
    simple_graph = create_simple_graph()
    advanced_graph = create_agent_graph()
    print("✅ 그래프 생성 완료\n")
    
    results = []
    
    for idx, question in enumerate(questions, 1):
        print(f"\n{'=' * 80}")
        print(f"테스트 {idx}/{len(questions)}: {question}")
        print(f"{'=' * 80}\n")
        
        print("🔹 기본 버전 실행 중...")
        start_time = time.time()
        simple_result = run_simple_agent(question, simple_graph)
        simple_time = time.time() - start_time
        print(f"✅ 기본 버전 완료 (소요 시간: {simple_time:.2f}초)\n")
        
        print("🔹 금융 특화 버전 실행 중...")
        start_time = time.time()
        advanced_result = run_agent(question, advanced_graph)
        advanced_time = time.time() - start_time
        print(f"✅ 금융 특화 버전 완료 (소요 시간: {advanced_time:.2f}초)\n")
        
        # 비교 리포트 출력
        compare_results(question, simple_result, advanced_result, simple_time, advanced_time)
        
        results.append({
            "question": question,
            "simple": simple_result,
            "advanced": advanced_result
        })
    
    # 전체 요약
    print("\n" + "=" * 80)
    print("📈 전체 테스트 요약")
    print("=" * 80)
    
    total_improvements = {
        "avg_doc_count_increase": 0,
        "avg_confidence": 0,
        "confidence_count": 0,
        "multi_search_count": 0,
        "verification_count": 0,
        "total_simple_time": 0,
        "total_advanced_time": 0,
        "total_simple_llm_calls": 0,
        "total_advanced_llm_calls": 0
    }
    
    for result in results:
        simple_time = result.get("simple_time", 0)
        advanced_time = result.get("advanced_time", 0)
        simple_metrics = extract_metrics(result["simple"], simple_time)
        advanced_metrics = extract_metrics(result["advanced"], advanced_time)
        
        total_improvements["avg_doc_count_increase"] += (advanced_metrics['doc_count'] - simple_metrics['doc_count'])
        if advanced_metrics['confidence_score'] is not None:
            total_improvements["avg_confidence"] += advanced_metrics['confidence_score']
            total_improvements["confidence_count"] += 1
        if advanced_metrics['search_round'] > 0:
            total_improvements["multi_search_count"] += 1
        if advanced_metrics['verification_round'] > 0:
            total_improvements["verification_count"] += 1
        if advanced_metrics.get('execution_time'):
            total_improvements["total_simple_time"] += simple_metrics.get('execution_time', 0)
            total_improvements["total_advanced_time"] += advanced_metrics['execution_time']
        if advanced_metrics.get('estimated_llm_calls'):
            total_improvements["total_simple_llm_calls"] += simple_metrics.get('estimated_llm_calls', 0)
            total_improvements["total_advanced_llm_calls"] += advanced_metrics['estimated_llm_calls']
    
    count = len(results)
    print(f"\n📊 성능 비교 요약")
    print("-" * 80)
    print(f"평균 문서 개수 증가: {total_improvements['avg_doc_count_increase'] / count:.2f}개")
    if total_improvements["confidence_count"] > 0:
        print(f"평균 신뢰도: {(total_improvements['avg_confidence'] / total_improvements['confidence_count'] * 100):.2f}% ({total_improvements['confidence_count']}개 질문 기준)")
    else:
        print("평균 신뢰도: 신뢰도 정보 없음")
    print(f"다중 검색 실행 비율: {total_improvements['multi_search_count']}/{count} ({total_improvements['multi_search_count']/count*100:.1f}%)")
    print(f"검증 실행 비율: {total_improvements['verification_count']}/{count} ({total_improvements['verification_count']/count*100:.1f}%)")
    
    if total_improvements["total_simple_time"] > 0:
        avg_simple_time = total_improvements["total_simple_time"] / count
        avg_advanced_time = total_improvements["total_advanced_time"] / count
        time_increase = ((avg_advanced_time - avg_simple_time) / avg_simple_time) * 100 if avg_simple_time > 0 else 0
        print(f"\n평균 실행 시간:")
        print(f"  - 기본 버전: {avg_simple_time:.2f}초")
        print(f"  - 금융 특화 버전: {avg_advanced_time:.2f}초")
        print(f"  - 시간 차이: {time_increase:+.1f}% ({avg_advanced_time - avg_simple_time:+.2f}초)")
    
    if total_improvements["total_simple_llm_calls"] > 0:
        avg_simple_llm = total_improvements["total_simple_llm_calls"] / count
        avg_advanced_llm = total_improvements["total_advanced_llm_calls"] / count
        print(f"\n평균 LLM 호출 횟수:")
        print(f"  - 기본 버전: {avg_simple_llm:.1f}회")
        print(f"  - 금융 특화 버전: {avg_advanced_llm:.1f}회")
        print(f"  - 차이: {avg_advanced_llm - avg_simple_llm:+.1f}회")
    
    print("\n" + "=" * 80)
    print(f"테스트 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    # 테스트 질문 리스트 (금융 관련 질문)
    test_questions = [
        "삼성전자 주가는 얼마인가요?",
        "한국의 기준금리는 몇 퍼센트인가요?",
        "비트코인 현재 가격은?",
    ]
    
    # 명령줄 인자로 질문 추가 가능
    if len(sys.argv) > 1:
        test_questions = sys.argv[1:]
    
    run_comparison(test_questions)

