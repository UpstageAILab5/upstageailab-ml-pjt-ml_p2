import streamlit as st
from review_crawling import (  # Import your predefined crawling functions
    get_movie_reviews_on_cgv,
    get_movie_reviews_on_megabox,
    CGV_URL,
    MEGABOX_URL,
    CGV_MOVIE_CODES,
    MEGABOX_MOVIE_CODES,
)
from nav import inject_custom_navbar
# Page configuration
st.set_page_config(
    page_title="영화 리뷰 크롤링",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_navbar()

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* General page styling */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #1C1C1E;
            color: #F0F0F0;
            font-family: 'Arial', sans-serif;
        }

        h1 {
            color: #FF416C;
            text-align: center;
            font-size: 36px;
            margin-bottom: 10px;
        }

        h2 {
            color: #FFFFFF;
            font-size: 28px;
            margin-top: 20px;
        }

        .stButton > button {
            background: linear-gradient(90deg, #FF416C, #FF4B2B);
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 10px rgba(255, 65, 108, 0.3);
        }

        .stButton > button:hover {
            background: linear-gradient(90deg, #FF4B2B, #FF416C);
            transform: scale(1.05);
        }

        .metric-box {
            background-color: #2C2C2E;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }

        .section-header {
            color: #28A745;
            font-size: 24px;
            margin-top: 30px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Page Title
st.title("🎥 영화 리뷰 크롤링")
st.write("이 페이지는 영화 리뷰를 크롤링하는 페이지입니다!")


def streamlit_movie_search():
    st.markdown("---")
    st.header("🔍 CGV 리뷰 크롤링")
    movie_list_cgv = list(CGV_MOVIE_CODES.keys())
    selected_movie_cgv = st.selectbox(
        "영화 선택 (CGV)",
        movie_list_cgv,
        index=None,
        placeholder="영화를 선택하세요...",
        key="cgv_selectbox",  # Unique key for CGV selectbox
    )
    cgv_review_limit = st.number_input("크롤링할 CGV 리뷰 수", min_value=1, value=5, key="cgv_limit")

    if st.button("CGV 리뷰 크롤링 시작", key="cgv_button"):
        if selected_movie_cgv:
            with st.spinner("리뷰 크롤링 중... 잠시만 기다려주세요."):
                cgv_url = CGV_URL + CGV_MOVIE_CODES[selected_movie_cgv]
                cgv_reviews = get_movie_reviews_on_cgv(url=cgv_url, review_limit=cgv_review_limit)

                st.success("리뷰 크롤링 완료!")
                st.subheader("CGV 리뷰")
                st.dataframe(cgv_reviews, use_container_width=True)

                st.download_button(
                    "CGV 리뷰 다운로드",
                    cgv_reviews.to_csv(index=False).encode("utf-8"),
                    f"cgv_reviews_{selected_movie_cgv}.csv",
                    "text/csv",
                )
        else:
            st.warning("영화를 먼저 선택하세요!")

    st.markdown("---")
    st.header("🔍 메가박스 리뷰 크롤링")
    movie_list_megabox = list(MEGABOX_MOVIE_CODES.keys())
    selected_movie_megabox = st.selectbox(
        "영화 선택 (메가박스)",
        movie_list_megabox,
        index=None,
        placeholder="영화를 선택하세요...",
        key="megabox_selectbox",  # Unique key for Megabox selectbox
    )
    megabox_review_limit = st.number_input("크롤링할 메가박스 리뷰 수", min_value=1, value=5, key="megabox_limit")

    if st.button("메가박스 리뷰 크롤링 시작", key="megabox_button"):
        if selected_movie_megabox:
            with st.spinner("리뷰 크롤링 중... 잠시만 기다려주세요."):
                megabox_url = MEGABOX_URL + MEGABOX_MOVIE_CODES[selected_movie_megabox]
                megabox_reviews = get_movie_reviews_on_megabox(url=megabox_url, review_limit=megabox_review_limit)

                st.success("리뷰 크롤링 완료!")
                st.subheader("메가박스 리뷰")
                st.dataframe(megabox_reviews, use_container_width=True)

                st.download_button(
                    "메가박스 리뷰 다운로드",
                    megabox_reviews.to_csv(index=False).encode("utf-8"),
                    f"megabox_reviews_{selected_movie_megabox}.csv",
                    "text/csv",
                )
        else:
            st.warning("영화를 먼저 선택하세요!")


if __name__ == "__main__":
    streamlit_movie_search()