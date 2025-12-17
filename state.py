"""
LangGraph RAG 에이전트의 상태 정의
"""
from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """
    그래프 상태 정의
    
    Attributes:
        messages: 대화 메시지 히스토리
        question: 현재 질문
        documents: 검색된 문서 리스트
        web_search: 웹 검색 필요 여부 (bool)
    """
    messages: Annotated[List, add_messages]
    question: str
    documents: List
    web_search: bool

