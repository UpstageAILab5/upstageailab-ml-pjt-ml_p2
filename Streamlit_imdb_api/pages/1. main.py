import streamlit as st

# Set page config
st.set_page_config(
    page_title="메인 페이지", 
    layout="wide", 
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None}, 
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
        /* Global page background and font */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #1C1C1E;
            color: #333;
            font-family: 'Arial', sans-serif;
        }

        /* Custom title styling */
        .title {
            text-align: center;
            color: #ff416c;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 10px;
        }

        /* Custom subtitle styling */
        .subtitle {
            text-align: center;
            color: #FFC300;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 20px;
        }

        /* Custom section headings */
        .section-header {
            text-align: center;
            color: #FFC300;
            font-size: 24px;
            margin-top: 20px;
        }

        /* Custom description styles */
        .description {
            text-align: center;
            color: #ECF0F1;
            font-size: 18px;
        }

        /* Custom emoji styles */
        .emoji {
            font-size: 24px;
            vertical-align: middle;
            margin-right: 10px;
        }

        /* Footer styling */
        footer {
            text-align: center;
            color: #555;
            font-size: 12px;
            margin-top: 50px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Page Content
st.markdown(
    '<h1 class="title">🎥 영화 리뷰 감성 분석 프로젝트</h1>', 
    unsafe_allow_html=True)
st.markdown(
    '<h2 class="subtitle">1️⃣ IMDB 50k 데이터를 활용한 영화 리뷰 기반 감성 분석</h2>', 
    unsafe_allow_html=True)
st.markdown(
    '<p class="description" style="text-align: center;">Model: <b>TinyBert/Albert</b></p>', 
    unsafe_allow_html=True)

st.markdown(
    '<h2 class="subtitle">2️⃣ CGV, 메가박스 영화 리뷰 기반 감성 분석</h2>', 
    unsafe_allow_html=True)
st.markdown(
    '<p class="description" style="text-align: center;">Model: <b>Kobert/WhitePeak</b></p>', 
    unsafe_allow_html=True)

st.markdown(
    '<h3 class="section-header">🛠️ Work Flow</h3>', 
    unsafe_allow_html=True)
st.markdown(
    '<p class="description" style="text-align: center;">사진 추가 예정</p>', 
    unsafe_allow_html=True)
# st.image("images/workflow.png", caption="Workflow Diagram")

# Footer
st.markdown('<footer>© 2024 오뚝이 프로젝트 팀 | Made with ❤️</footer>', unsafe_allow_html=True)