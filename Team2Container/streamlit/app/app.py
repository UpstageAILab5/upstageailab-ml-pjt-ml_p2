import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from library.s3 import s3_client

from menu.home import home_view
from menu.text_analysis import text_analysis_view
from menu.movie_review_analysis import review_analysis

# 페이지 설정과 스타일링
st.set_page_config(page_title="영화 리뷰 감성 분석", layout="wide")
st.markdown(
    """
    <style>
        #fce8dc1b {
            text-align: center;
        }
        [data-testid="stMainBlockContainer"] {
            width: 80%;
            margin: auto;
        }
        .stHorizontalBlock st-emotion-cache-ocqkz7 e1f1d6gn5 {
            display: flex;
            width: 100%;
        }
        .stColumn st-emotion-cache-1r6slb0 e1f1d6gn3 {
            flex: 1;
        }
        /* 탭 레이블의 글씨 크기 조절 */
        [data-baseweb="tab"] [data-testid="stMarkdownContainer"] p {
            font-size: 30px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 데이터 준비
images = s3_client(st.secrets['access_key'], st.secrets['secret_access_key']).get_images()
detail_images = [
    "https://fastcampus-ml-p2-bucket.s3.ap-northeast-2.amazonaws.com/image/detail_image1.avif"
]
titles = [
    "Moana2",
    "wicked",
    "Gladiator2"
]
buttons = []

# 메인 페이지 시작

# 메뉴 옵션 정의
menu_options = ["🏠 홈", "📝 텍스트 분석", "🎞️ 영화 리뷰 분석"]
tab1, tab2, tab3 = st.tabs(menu_options)

# 현재 선택된 메뉴에 따라 콘텐츠 표시
with tab1:
    home_view(titles, images, buttons, detail_images)
with tab2:
    text_analysis_view()
with tab3:
    review_analysis()
