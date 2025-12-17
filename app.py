"""
Streamlit 기반 RAG 에이전트 UI
"""
import streamlit as st
from main import create_agent_graph, run_agent
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="LangGraph RAG 에이전트",
    page_icon="🤖",
    layout="wide"
)

# 제목
st.title("🤖 LangGraph RAG 에이전트")
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
    
    if not openai_key:
        st.error("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
    else:
        st.success("✅ OpenAI API 키 설정됨")
    
    if not tavily_key:
        st.warning("⚠️ TAVILY_API_KEY가 설정되지 않았습니다. 웹 검색 기능이 작동하지 않을 수 있습니다.")
    else:
        st.success("✅ Tavily API 키 설정됨")
    
    st.markdown("---")
    st.markdown("### 📚 사용 방법")
    st.markdown("""
    1. 아래 입력창에 질문을 입력하세요
    2. Enter를 누르거나 전송 버튼을 클릭하세요
    3. 에이전트가 벡터 스토어와 웹을 검색하여 답변을 생성합니다
    """)

# 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 에이전트 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하는 중..."):
            result = run_agent(prompt, st.session_state.graph)
            
            # 최종 답변 추출
            if result.get("messages"):
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    answer = last_message.content
                else:
                    answer = str(last_message)
            else:
                answer = "답변을 생성할 수 없습니다."
            
            st.markdown(answer)
    
    # 어시스턴트 메시지 추가
    if result.get("messages"):
        last_message = result["messages"][-1]
        if hasattr(last_message, 'content'):
            answer_content = last_message.content
        else:
            answer_content = str(last_message)
        st.session_state.messages.append({"role": "assistant", "content": answer_content})

# 하단 정보
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>LangGraph 기반 RAG 에이전트 | Powered by OpenAI & Tavily</p>
    </div>
    """,
    unsafe_allow_html=True
)

