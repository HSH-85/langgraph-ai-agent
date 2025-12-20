"""
Streamlit 기반 고도화된 RAG 에이전트 UI
"""
import streamlit as st
from main import create_agent_graph, run_agent
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="고도화된 LangGraph RAG 에이전트",
    page_icon="🤖",
    layout="wide"
)

# 제목
st.title("💰 금융 특화 LangGraph RAG 에이전트")
st.markdown("*금융 도메인 분석 · 다중 검색/검증 · 신뢰도 기반 답변 생성*")
st.markdown("---")

# 세션 상태 초기화
if "graph" not in st.session_state:
    st.session_state.graph = create_agent_graph()
    st.session_state.messages = []

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    
    # API 키 확인
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")
    
    if not openai_key:
        st.error("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
    else:
        st.success("✅ OpenAI API 키 설정됨")
    
    if not tavily_key:
        st.warning("⚠️ TAVILY_API_KEY가 설정되지 않았습니다. 웹 검색 기능이 작동하지 않을 수 있습니다.")
    else:
        st.success("✅ Tavily API 키 설정됨")
    
    if not cohere_key:
        st.warning("⚠️ COHERE_API_KEY가 설정되지 않았습니다. 리랭크 기능이 작동하지 않을 수 있습니다.")
    else:
        st.success("✅ Cohere API 키 설정됨")
    
    st.markdown("---")
    st.markdown("### 📚 사용 방법")
    st.markdown("""
    1. 아래 입력창에 질문을 입력하세요
    2. Enter를 누르거나 전송 버튼을 클릭하세요
    3. 에이전트가 자동으로:
       - 질문 의도 분석
       - 문서 검색 및 리랭크
       - 관련성 평가 및 웹 검색
       - 최적화된 답변 생성
    """)
    
    # 고급 옵션 토글
    show_thought = st.checkbox("🧠 사고 과정 표시", value=True)
    show_meta = st.checkbox("📊 메타 정보 표시", value=False)

# 채팅 히스토리 표시
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # 메타 정보가 있으면 표시
        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]
            
            # 사고 과정 표시
            if show_thought and "thought_process" in metadata:
                with st.expander("🧠 사고 과정 보기"):
                    for thought in metadata["thought_process"]:
                        st.caption(thought)
            
            # 메타 정보 표시 (금융 특화)
            if show_meta:
                with st.expander("📊 메타 정보 보기"):
                    # 기본 정보
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("의도", metadata.get("intent", "N/A"))
                    with col2:
                        st.metric("관련성", metadata.get("is_relevant", "N/A"))
                    with col3:
                        st.metric("문서 수", metadata.get("doc_count", 0))
                    
                    # 금융 특화 정보
                    domain_kr = {
                        'stock': '주식', 'bond': '채권', 'forex': '외환',
                        'real_estate': '부동산', 'interest_rate': '금리',
                        'derivative': '파생상품', 'crypto': '암호화폐',
                        'economic': '경제 지표', 'general': '일반 금융'
                    }
                    financial_domain = metadata.get("financial_domain")
                    if financial_domain:
                        st.markdown(f"**💰 금융 도메인**: {domain_kr.get(financial_domain, financial_domain)}")
                    
                    confidence_score = metadata.get("confidence_score")
                    if confidence_score is not None:
                        st.markdown(f"**📊 신뢰도**: {confidence_score:.2%}")
                        st.progress(confidence_score)
                    
                    source_agreement = metadata.get("source_agreement")
                    if source_agreement:
                        agreement_kr = {"high": "높음", "medium": "보통", "low": "낮음"}
                        st.markdown(f"**🔄 소스 일치도**: {agreement_kr.get(source_agreement, source_agreement)}")
                    
                    col4, col5 = st.columns(2)
                    with col4:
                        st.caption(f"🔍 검색 라운드: {metadata.get('search_round', 0)}")
                    with col5:
                        st.caption(f"✅ 검증 라운드: {metadata.get('verification_round', 0)}")
                    
                    if metadata.get("loop_count", 0) > 0:
                        st.caption(f"🔄 재시도 횟수: {metadata['loop_count']}")
                    if metadata.get("web_search_used"):
                        st.caption("🌐 웹 검색 사용됨")

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 에이전트 응답 생성
    with st.chat_message("assistant"):
        # 진행 상태 표시
        status_placeholder = st.empty()
        
        with st.spinner("답변을 생성하는 중..."):
            # 이전 메시지를 BaseMessage 형식으로 변환 (현재 메시지 제외)
            from langchain_core.messages import HumanMessage, AIMessage
            previous_messages = []
            # 현재 메시지를 제외한 이전 메시지만 변환
            for msg in st.session_state.messages[:-1]:  # 마지막 메시지(현재 사용자 메시지) 제외
                if msg["role"] == "user":
                    previous_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    previous_messages.append(AIMessage(content=msg["content"]))
            
            try:
                result = run_agent(prompt, st.session_state.graph, previous_messages=previous_messages)
            except Exception as e:
                st.error(f"에러가 발생했습니다: {str(e)}")
                st.exception(e)
                result = None
            
            # result가 None이면 에러 메시지 표시 후 종료
            if result is None:
                status_placeholder.empty()
                st.error("답변을 생성할 수 없습니다. 에러 로그를 확인해주세요.")
                st.stop()
            
            # 최종 답변 추출
            if result.get("messages"):
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    answer = last_message.content
                else:
                    answer = str(last_message)
            else:
                answer = "답변을 생성할 수 없습니다."
            
            status_placeholder.empty()
            st.markdown(answer)
            
            # 메타데이터 수집 (금융 특화 필드 포함)
            metadata = {
                "intent": result.get("intent"),
                "is_relevant": result.get("is_relevant"),
                "doc_count": len(result.get("documents", [])),
                "loop_count": result.get("loop_count", 0),
                "web_search_used": result.get("loop_count", 0) > 0,
                "thought_process": result.get("thought_process", []),
                # 금융 특화 필드
                "financial_domain": result.get("financial_domain"),
                "confidence_score": result.get("confidence_score"),
                "source_agreement": result.get("source_agreement"),
                "search_round": result.get("search_round", 0),
                "verification_round": result.get("verification_round", 0)
            }
            
            # 사고 과정 표시
            if show_thought and metadata["thought_process"]:
                with st.expander("🧠 사고 과정 보기"):
                    for thought in metadata["thought_process"]:
                        st.caption(thought)
            
            # 메타 정보 표시 (금융 특화)
            if show_meta:
                with st.expander("📊 메타 정보 보기"):
                    # 기본 정보
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("의도", metadata.get("intent", "N/A"))
                    with col2:
                        st.metric("관련성", metadata.get("is_relevant", "N/A"))
                    with col3:
                        st.metric("문서 수", metadata.get("doc_count", 0))
                    
                    # 금융 특화 정보
                    domain_kr = {
                        'stock': '주식', 'bond': '채권', 'forex': '외환',
                        'real_estate': '부동산', 'interest_rate': '금리',
                        'derivative': '파생상품', 'crypto': '암호화폐',
                        'economic': '경제 지표', 'general': '일반 금융'
                    }
                    financial_domain = metadata.get("financial_domain")
                    if financial_domain:
                        st.markdown(f"**💰 금융 도메인**: {domain_kr.get(financial_domain, financial_domain)}")
                    
                    confidence_score = metadata.get("confidence_score")
                    if confidence_score is not None:
                        st.markdown(f"**📊 신뢰도**: {confidence_score:.2%}")
                        st.progress(confidence_score)
                    
                    source_agreement = metadata.get("source_agreement")
                    if source_agreement:
                        agreement_kr = {"high": "높음", "medium": "보통", "low": "낮음"}
                        st.markdown(f"**🔄 소스 일치도**: {agreement_kr.get(source_agreement, source_agreement)}")
                    
                    col4, col5 = st.columns(2)
                    with col4:
                        st.caption(f"🔍 검색 라운드: {metadata.get('search_round', 0)}")
                    with col5:
                        st.caption(f"✅ 검증 라운드: {metadata.get('verification_round', 0)}")
                    
                    if metadata.get("loop_count", 0) > 0:
                        st.caption(f"🔄 재시도 횟수: {metadata['loop_count']}")
                    if metadata.get("web_search_used"):
                        st.caption("🌐 웹 검색 사용됨")
    
    # 어시스턴트 메시지 추가 (메타데이터 포함)
    if result and result.get("messages"):
        last_message = result["messages"][-1]
        if hasattr(last_message, 'content'):
            answer_content = last_message.content
        else:
            answer_content = str(last_message)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer_content,
            "metadata": metadata
        })

# 하단 정보
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>💰 금융 특화 LangGraph RAG 에이전트 | Powered by OpenAI, Tavily & Cohere</p>
        <p style='font-size: 0.8em;'>금융 도메인 분석 · 다중 검색/검증 · 신뢰도 기반 답변 · 사고 과정 추적</p>
    </div>
    """,
    unsafe_allow_html=True
)

