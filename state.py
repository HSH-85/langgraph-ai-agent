"""
LangGraph RAG 에이전트의 고도화된 상태 정의
"""
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    고도화된 RAG 에이전트의 상태 관리 스키마
    
    기존 GraphState를 확장하여 다음 기능 추가:
    - 사용자 의도 분석 (intent)
    - 루프 카운터로 무한 루프 방지
    - 사고 과정 추적 (thought_process)
    - 컨텍스트 관리 개선
    """
    
    # ========== 1. 대화 기록 ==========
    messages: Annotated[List[BaseMessage], add_messages]
    """대화 메시지 히스토리 (자동 누적)"""
    
    # ========== 2. 사용자 입력 및 분석 ==========
    question: str
    """사용자의 최종 질문 (정제된 형태)"""
    
    intent: Optional[str]
    """사용자의 의도 분석 결과
    - 'factual': 사실적 정보 검색
    - 'analytical': 분석/비교 요구
    - 'conversational': 일반 대화
    - 'procedural': 절차/방법 설명
    """
    
    # ========== 3. 데이터 및 지식 베이스 ==========
    documents: List[str]
    """Retriever나 Web Search로 수집된 문서들"""
    
    context: Optional[str]
    """최종 답변 생성용 정리된 컨텍스트"""
    
    # ========== 4. 품질 관리 및 제어 플래그 ==========
    is_relevant: Optional[str]
    """문서 관련성 평가 결과
    - 'yes': 관련성 높음
    - 'no': 관련성 없음
    - 'partial': 부분적으로 관련
    """
    
    web_search: bool
    """웹 검색 필요 여부 (기존 web_search와 호환)"""
    
    loop_count: int
    """무한 루프 방지용 재시도 카운터 (최대 3회)"""
    
    # ========== 5. UI/UX 및 모니터링 ==========
    current_step: Optional[str]
    """현재 에이전트 실행 단계 (UI 표시용)
    예: '의도 분석 중...', '문서 검색 중...', '답변 생성 중...'
    """
    
    thought_process: List[str]
    """에이전트의 내부 사고 로그 (투명성 제공)"""


# 하위 호환성을 위한 별칭 유지
GraphState = AgentState