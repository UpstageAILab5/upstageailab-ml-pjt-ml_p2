import streamlit as st
import pandas as pd
from review_crawling import (                       # 미리 만들은 크롤링 함수를 불러옴
    get_movie_reviews_on_cgv,
    get_movie_reviews_on_megabox,
    CGV_URL,
    MEGABOX_URL,
    CGV_MOVIE_CODES,
    MEGABOX_MOVIE_CODES
)


###################### 페이지 설정 ######################
# 사이드바 설정 (여기에 def 페이지 함수를 추가하세요)
# 그리고 각 페이지 아래에 기능을 추가하고 싶으면 'def crawling_page()' 를 참고하세요
st.sidebar.title("Contents")
page = st.sidebar.radio("원하시는 페이지를 선택하세요", ["홈", "소개", "크롤링"])  # 추가한 페이지에 맞게 수정

def home_page():
    st.title("2조 오뚝이")
    st.write("페이지에 오신 것을 환영합니다!")

def about_page():
    st.title("소개")
    st.write("이 페이지는 2조 오뚝이의 영화 리뷰 페이지입니다!")

def crawling_page():
    st.title("크롤링")
    st.write("이 페이지는 영화 리뷰를 크롤링하는 페이지입니다!")
    streamlit_movie_search()                                            # 아래에 정의한 기능 추가 

def sentiment_analysis_page():
    st.title("감성 분석")
    st.write("이 페이지는 영화 리뷰를 감성 분석하는 페이지입니다!")


###################### 추가할 기능 설정 ######################
# 추가할 기능을 함수로 설정하세요 

# 크롤링 기능 설정
def streamlit_movie_search():
    st.title("영화 리뷰 크롤링")
    if st.checkbox("CGV 리뷰 크롤링"):  
        def cgv_review_crawling():
            st.write("CGV 영화 리뷰 크롤링")

        movie_list = list(CGV_MOVIE_CODES.keys())
        selected_movie = st.selectbox(
            "영화 선택",                                      # 영화 선택 텍스트
            movie_list,                                     # 영화 선택 리스트
            index=None,                                     # 영화 선택 시 초기 선택 값 없음
            placeholder="영화를 선택하세요..."                  # 영화 선택 시 플레이스홀더 텍스트
        )
        cgv_review_limit = st.number_input("크롤링할 CGV 리뷰 수", min_value=1, value=5)

        if st.button("리뷰 크롤링"):
            if selected_movie:
                with st.spinner("리뷰 크롤링 중... 잠시만 기다려주세요."):
                    cgv_url = CGV_URL + CGV_MOVIE_CODES[selected_movie]
                    cgv_reviews = get_movie_reviews_on_cgv(url=cgv_url, review_limit=cgv_review_limit)

                    st.subheader("CGV 리뷰")
                    st.dataframe(cgv_reviews, use_container_width=True)

                    st.download_button(
                        "CGV 리뷰 다운로드",
                        cgv_reviews.to_csv(index=False).encode('utf-8'),
                        f"cgv_reviews_{selected_movie}.csv",
                        "text/csv"
                    )
            else:
                st.warning("영화를 먼저 선택하세요!")
    
    if st.checkbox("메가박스 리뷰 크롤링"):
        def megabox_review_crawling():
            st.write("메가박스 영화 리뷰 크롤링")

        movie_list = list(MEGABOX_MOVIE_CODES.keys())
        selected_movie = st.selectbox(
            "영화 선택",
            movie_list,
            index=None,
            placeholder="영화를 선택하세요..."
        )
        megabox_review_limit = st.number_input("크롤링할 메가박스 리뷰 수", min_value=1, value=5)

        if st.button("리뷰 크롤링"):
            if selected_movie:
                with st.spinner("리뷰 크롤링 중... 잠시만 기다려주세요."):
                    megabox_url = MEGABOX_URL + MEGABOX_MOVIE_CODES[selected_movie]
                    megabox_reviews = get_movie_reviews_on_megabox(url=megabox_url, review_limit=megabox_review_limit)

                    st.subheader("메가박스 리뷰")
                    st.dataframe(megabox_reviews, use_container_width=True)

                    st.download_button(
                        "메가박스 리뷰 다운로드",
                        megabox_reviews.to_csv(index=False).encode('utf-8'),
                        f"megabox_reviews_{selected_movie}.csv",
                        "text/csv"
                    )
            else:
                st.warning("영화를 먼저 선택하세요!")

# 분류 기능 설정
def sentiment_analysis_page():
    st.title("감성 분석")
    st.write("이 페이지는 영화 리뷰를 감성 분석하는 페이지입니다!")


# 페이지 추가 시 함께 수정해주세요 
if page == "홈":
    home_page()
elif page == "소개":
    about_page()
elif page == "크롤링":
    crawling_page()
elif page == "감성 분석":
    sentiment_analysis_page()


if __name__ == "__main__":

    pass



    # st.write("페이지를 선택하세요")
    # home_page()
    # about_page()
    # crawling_page()
    # # streamlit_movie_search() 